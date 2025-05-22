import os
import glob
import numpy as np
import zarr
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


data_transforms = T.Compose([
    T.Resize((240, 320)),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.RandomResizedCrop((240, 320), scale=(0.9, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


class RobotArmDataset(Dataset):
    def __init__(self, zarr_path, frames_root, transform):

        zroot = zarr.open(zarr_path, mode='r')
        self.pos         = zroot['data/robot_eef_pos'][:]   # (N,3)
        self.quat        = zroot['data/robot_eef_quat'][:]  # (N,4)
        ep_ends          = zroot['meta/episode_ends'][:]    # (num_eps,)

        self.bounds = []
        start = 0
        for end in ep_ends:
            self.bounds.append((start, end))
            start = end
        self.total_frames = self.pos.shape[0]

        self.hand_paths  = []
        self.third_paths = []
        for ep_idx, (st, en) in enumerate(self.bounds):
            hp = sorted(glob.glob(os.path.join(frames_root, str(ep_idx), 'hand',  '*.png')))
            tp = sorted(glob.glob(os.path.join(frames_root, str(ep_idx), 'third', '*.png')))
            self.hand_paths  += hp[:(en-st)]
            self.third_paths += tp[:(en-st)]

        self.transform = transform

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        hand  = Image.open(self.hand_paths[idx]).convert('RGB')
        third = Image.open(self.third_paths[idx]).convert('RGB')

        hand_v = self.transform(hand)
        third_v = self.transform(third)

        pose = torch.tensor(np.concatenate([self.pos[idx], self.quat[idx]]),
                            dtype=torch.float)

        view = {'hand_img': hand_v, 'third_img': third_v, 'pose': pose}
        return view, view


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, x):
        return self.cnn(x)

class FusionEncoder(nn.Module):
    def __init__(self, img_embed=128, final_embed=128):
        super().__init__()
        self.image_encoder = ImageEncoder(img_embed)
        # self.fc = nn.Sequential(
        #     nn.Linear(img_embed * 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, final_embed)
        # )
        self.fc = nn.Sequential(
            nn.Linear(img_embed, 256),
            nn.ReLU(),
            nn.Linear(256, final_embed)
        )

    def forward(self, hand_img, third_img, pose=None):
        # h = self.image_encoder(hand_img)
        t = self.image_encoder(third_img)
        # x = torch.cat([h, t], dim=1)
        x = t
        e = self.fc(x)
        return e / e.norm(dim=1, keepdim=True)



if __name__ == "__main__":
    zarr_path   = 'multimodal_training/replay_buffer.zarr'
    frames_root = 'multimodal_training/frames'
    save_path   = 'traj_database.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionEncoder().to(device)
    model.load_state_dict(torch.load("fusion_encoder_contrastive_fast.pth", map_location=device))
    model.eval()

    dataset = RobotArmDataset(zarr_path=zarr_path, frames_root=frames_root, transform=data_transforms)

    all_embeddings = []
    info_list = []

    for idx in range(len(dataset)):
        (view, _ ) = dataset[idx]
        h = view['hand_img'].unsqueeze(0).to(device)
        t = view['third_img'].unsqueeze(0).to(device)
        p = view['pose'].unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(h, t, p).squeeze(0).cpu()
            all_embeddings.append(emb)

        for ep_idx, (st, en) in enumerate(dataset.bounds):
            if idx < en:
                info_list.append((ep_idx, idx - st))
                break

        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1}/{len(dataset)} frames")

    embeddings_tensor = torch.stack(all_embeddings)
    traj_database = {'embeddings': embeddings_tensor, 'info': info_list}
    torch.save(traj_database, save_path)
    print(f"Saved trajectory database to {save_path}")

