import os
import cv2
import numpy as np
import zarr
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# data preprocessing
data_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),  # the resolution of the image input should be 320x240
    # T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.RandomResizedCrop((240, 320), scale=(0.9, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# Same as the training script
class RobotArmDataset(Dataset):
    def __init__(self, zarr_path, videos_dir, transform):
        """
        zarr_path: zarr path
        videos_dir: video path
        transform: the image preprocessing and augmentation pipeline
        """
        self.transform = transform
        self.videos_dir = videos_dir

        # load zarr data
        zarr_root = zarr.open(zarr_path, mode='r')
        self.robot_eef_pos = zarr_root['data/robot_eef_pos'][:]  # (N, 3)
        self.robot_eef_quat = zarr_root['data/robot_eef_quat'][:]  # (N, 4)
        # the end index of each episode - same as diffusion policy
        self.episode_ends = zarr_root['meta/episode_ends'][:]

        # access the episode boundaries to get the start and end index of each episode and the global index
        self.episode_boundaries = []
        start = 0
        for end in self.episode_ends:
            self.episode_boundaries.append((start, end))
            start = end
        self.total_frames = self.robot_eef_pos.shape[0]

        # preload video frames - we need to change this part completely when we transfer to larger dataset
        self.video_frames = {}
        num_eps = len(self.episode_boundaries)
        for ep in range(num_eps):
            ep_dir = os.path.join(videos_dir, str(ep))
            hand_video_path = os.path.join(ep_dir, "1.mp4")
            third_video_path = os.path.join(ep_dir, "3.mp4")
            hand_frames = self.load_video_frames(hand_video_path)
            third_frames = self.load_video_frames(third_video_path)
            self.video_frames[ep] = {'hand': hand_frames, 'third': third_frames}

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 将 BGR 转为 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # access the global index and get the corresponding episode and local index
        ep = 0
        local_idx = 0
        for i, (start, end) in enumerate(self.episode_boundaries):
            if idx < end:
                ep = i
                local_idx = idx - start
                break

        # find the corresponding video frames by local index
        hand_img = self.video_frames[ep]['hand'][local_idx]  # numpy array, H x W x 3
        third_img = self.video_frames[ep]['third'][local_idx]

        # get eef info from zarr
        pos = self.robot_eef_pos[idx]
        quat = self.robot_eef_quat[idx]
        pose = np.concatenate([pos, quat], axis=0)

        # use view1 to build the latent embedding database
        hand_img_v1 = self.transform(hand_img.copy())
        third_img_v1 = self.transform(third_img.copy())
        # for pose,
        pose_tensor = torch.tensor(pose, dtype=torch.float)

        view1 = {'hand_img': hand_img_v1, 'third_img': third_img_v1, 'pose': pose_tensor}
        # return view1 only
        return view1, view1


# define the model - should be the same as the training script
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, x):
        return self.cnn(x)


class PoseEncoder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FusionEncoder(nn.Module):
    def __init__(self, img_embed=128, pose_embed=32, final_embed=128):
        super().__init__()
        self.image_encoder = ImageEncoder(img_embed)
        # third_img + hand_img + pose
        # self.pose_encoder = PoseEncoder(pose_embed)
        # self.fc = nn.Sequential(
        #     nn.Linear(img_embed * 2 + pose_embed, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, final_embed)
        # )
        # third_img + hand_img
        self.fc = nn.Sequential(
            nn.Linear(img_embed * 2, 256),
            nn.ReLU(),
            nn.Linear(256, final_embed)
        )
        # third_img
        # self.fc = nn.Sequential(
        #     nn.Linear(img_embed, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, final_embed)
        # )

    def forward(self, hand_img, third_img, pose):
        """
        Three modes again:
        1. hand_img + third_img + pose
        2. hand_img + third_img
        3. third_img
        """
        # hand_img, third_img: [B, 3, H, W]，pose: [B, 7]
        hand_feat = self.image_encoder(hand_img)  # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        # pose_feat = self.pose_encoder(pose)  # [B, pose_embed]
        # fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        fused = torch.cat([hand_feat, third_feat], dim=1)
        # fused = third_feat
        embedding = self.fc(fused)
        # L2 normalization
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding


# build the trajectory database
if __name__ == "__main__":
    # path to the zarr file and video frames
    zarr_path = 'rgb_training/replay_buffer.zarr'
    videos_dir = 'rgb_training/videos'
    save_path = 'traj_database.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the trained model - MAKE SURE THE MODEL IS THE SAME AS THE TRAINING SCRIPT
    model = FusionEncoder().to(device)
    model.load_state_dict(torch.load("fusion_encoder_contrastive.pth", map_location=device))
    model.eval()

    # load the dataset
    dataset = RobotArmDataset(zarr_path=zarr_path, videos_dir=videos_dir, transform=data_transforms)

    all_embeddings = []
    info_list = []

    # iterating over the dataset
    for idx in range(len(dataset)):
        view1, _ = dataset[idx]  # only view1 is used to build the database
        # there are three information in view1
        hand_img = view1['hand_img'].unsqueeze(0).to(device)
        third_img = view1['third_img'].unsqueeze(0).to(device)
        pose = view1['pose'].unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(hand_img, third_img, pose)
            emb = emb.squeeze(0).cpu()
            all_embeddings.append(emb)

        # get the episode index and local index
        for ep_idx, (start, end) in enumerate(dataset.episode_boundaries):
            if idx < end:
                local_idx = idx - start
                info_list.append((ep_idx, local_idx))
                break

        if (idx + 1) % 100 == 0:
            print(f"deal with {idx + 1} / {len(dataset)} frames")

    # concatenate all embeddings, resulting shape: [N, final_embed]
    embeddings_tensor = torch.stack(all_embeddings)

    # build the trajectory database, which is a dictionary containing the embeddings and the info list
    traj_database = {'embeddings': embeddings_tensor, 'info': info_list}
    torch.save(traj_database, save_path)
    print(f"traj_database is saved in {save_path}")
