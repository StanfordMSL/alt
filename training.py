import os
import cv2
import numpy as np
import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# define the data augmentation pipeline
# First, we convert the image to a PIL image
# then resize it to 320x240 (note that the order of dimensions in PIL is (height, width) so we write (240,320))
# finally we apply random horizontal flip, color jitter, and finally convert it to a tensor and normalize it.
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

# create different views of the image with random augmentations (different each time)
class RobotArmDataset(Dataset):
    def __init__(self, zarr_path, videos_dir, transform):
        """
        zarr_path: low_dim data path
        videos_dir: image data path
        transform: preprocessing pipeline
        """
        self.transform = transform
        self.videos_dir = videos_dir

        # load low-dim data from zarr
        zarr_root = zarr.open(zarr_path, mode='r')
        self.robot_eef_pos = zarr_root['data/robot_eef_pos'][:]   # (N, 3)
        self.robot_eef_quat = zarr_root['data/robot_eef_quat'][:]  # (N, 4)
        self.episode_ends = zarr_root['meta/episode_ends'][:]      # (num_episode,)

        # according to episode_ends, we can get the boundaries of each episode
        self.episode_boundaries = []
        start = 0
        for end in self.episode_ends:
            self.episode_boundaries.append((start, end))
            start = end
        self.total_frames = self.robot_eef_pos.shape[0]

        # loading video frames
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
            # BGR->RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # according to the global index, we can get the episode_id and local index
        ep = 0
        local_idx = 0
        for i, (start, end) in enumerate(self.episode_boundaries):
            if idx < end:
                ep = i
                local_idx = idx - start
                break

        # according to the episode and local index, we can get the image data
        hand_img = self.video_frames[ep]['hand'][local_idx]    # numpy array, H x W x 3
        third_img = self.video_frames[ep]['third'][local_idx]

        # get access the low-dim data
        pos = self.robot_eef_pos[idx]   # (3,)
        quat = self.robot_eef_quat[idx] # (4,)
        pose = np.concatenate([pos, quat], axis=0)  # (7,)

        # create different views of the image with random augmentations (different each time)
        # do random augmentations for each view
        hand_img_v1 = self.transform(hand_img.copy())
        third_img_v1 = self.transform(third_img.copy())
        hand_img_v2 = self.transform(hand_img.copy())
        third_img_v2 = self.transform(third_img.copy())

        # add some noise to the low-dim data
        pose_tensor = torch.tensor(pose, dtype=torch.float)
        noise1 = torch.randn_like(pose_tensor) * 0.01
        noise2 = torch.randn_like(pose_tensor) * 0.01
        pose_v1 = pose_tensor + noise1
        pose_v2 = pose_tensor + noise2

        # define two views for contrastive learning
        view1 = {'hand_img': hand_img_v1, 'third_img': third_img_v1, 'pose': pose_v1}
        view2 = {'hand_img': hand_img_v2, 'third_img': third_img_v2, 'pose': pose_v2}

        return view1, view2

# Define the encoder
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, x):
        return self.cnn(x)

# class PoseEncoder(nn.Module):
#     def __init__(self, embed_dim=32):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(7, 64),
#             nn.ReLU(),
#             nn.Linear(64, embed_dim)
#         )
#     def forward(self, x):
#         return self.mlp(x)

class FusionEncoder(nn.Module):
    def __init__(self, img_embed=128, pose_embed=32, final_embed=128):
        super().__init__()
        self.image_encoder = ImageEncoder(img_embed)
        # use pos + third_img + first_img
        # self.pose_encoder = PoseEncoder(pose_embed)
        # self.fc = nn.Sequential(
        #     nn.Linear(img_embed * 2 + pose_embed, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, final_embed)
        # use hand_img + third_img
        self.fc = nn.Sequential(
                nn.Linear(img_embed * 2, 256),
                nn.ReLU(),
                nn.Linear(256, final_embed)
        )
        # only third view is used
        # self.fc = nn.Sequential(
        #     nn.Linear(img_embed, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, final_embed)
        # )
    def forward(self, hand_img, third_img, pose):
        """
        we have three modes:
            1. hand_img + third_img + pose
            2. hand_img + third_img
            3. third_img
        """
        # hand_img, third_img: [B, 3, H, W]，pose: [B, 7]
        hand_feat = self.image_encoder(hand_img)   # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        # pose_feat = self.pose_encoder(pose)          # [B, pose_embed]
        # fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        fused = torch.cat([hand_feat, third_feat], dim=1)
        # fused = third_feat
        embedding = self.fc(fused)
        # use L2 normalization
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding

# contrastive loss
# we use NTXentLoss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        return mask

    def forward(self, z_i, z_j):
        # z_i, z_j: [batch_size, embed_dim]
        N = 2 * self.batch_size
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embed_dim]
        sim = torch.matmul(z, z.T) / self.temperature  # similarity matirx [2B, 2B]

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i]).reshape(N, 1)
        negatives = sim[self.mask].reshape(N, -1)
        # build the labels
        labels = torch.zeros(N, dtype=torch.long).to(self.device)
        # build the logits
        logits = torch.cat([positives, negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# Training the model
# Hyperparameters
batch_size = 32
num_epochs = 40    # 40 as default
learning_rate = 1e-4
temperature = 0.4  # 0.4 as default
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data path of videos and zarr file
zarr_path = 'rgb_training/replay_buffer.zarr'
videos_dir = 'rgb_training/videos'

dataset = RobotArmDataset(zarr_path=zarr_path, videos_dir=videos_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

# initialize the model, optimizer and loss function
model = FusionEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
nt_xent_loss = NTXentLoss(batch_size=batch_size, temperature=temperature, device=device)

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        # each batch contains two views
        view1, view2 = batch
        hand_img1 = view1['hand_img'].to(device)
        third_img1 = view1['third_img'].to(device)
        pose1 = view1['pose'].to(device)
        hand_img2 = view2['hand_img'].to(device)
        third_img2 = view2['third_img'].to(device)
        pose2 = view2['pose'].to(device)

        # get two embeddings
        z1 = model(hand_img1, third_img1, pose1)  # [B, final_embed]
        z2 = model(hand_img2, third_img2, pose2)  # [B, final_embed]

        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# save the trained model
torch.save(model.state_dict(), "fusion_encoder_contrastive.pth")
