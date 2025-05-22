import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

torch.backends.cudnn.benchmark = True


class RobotArmDataset(Dataset):
    def __init__(self, zarr_path, frames_root, transform):
        import zarr
        zroot = zarr.open(zarr_path, mode="r")
        self.pos     = zroot['data/robot_eef_pos'][:]
        self.quat    = zroot['data/robot_eef_quat'][:]
        self.ep_ends = zroot['meta/episode_ends'][:]

        self.bounds = []
        start = 0
        for end in self.ep_ends:
            self.bounds.append((start, end))
            start = end
        self.total = self.pos.shape[0]

        self.hand_paths  = []
        self.third_paths = []
        for ep_idx, (st, en) in enumerate(self.bounds):
            hp = sorted(glob.glob(os.path.join(frames_root, str(ep_idx), "hand",  "*.png")))
            tp = sorted(glob.glob(os.path.join(frames_root, str(ep_idx), "third", "*.png")))

            self.hand_paths  += hp[:(en-st)]
            self.third_paths += tp[:(en-st)]

        self.transform = transform

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        hand  = Image.open(self.hand_paths[idx]).convert("RGB")
        third = Image.open(self.third_paths[idx]).convert("RGB")
        h1, h2 = self.transform(hand),  self.transform(hand)
        t1, t2 = self.transform(third), self.transform(third)
        p = torch.tensor(np.concatenate([self.pos[idx], self.quat[idx]]), dtype=torch.float)
        noise = lambda: torch.randn_like(p) * 0.01
        return (
            {'hand_img':h1,'third_img':t1,'pose':p+noise()},
            {'hand_img':h2,'third_img':t2,'pose':p+noise()},
        )


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_f = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_f, embed_dim)

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

# contrastive loss
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super().__init__()
        self.bs   = batch_size
        self.temp = temperature
        self.device = device
        self.crit = nn.CrossEntropyLoss(reduction="sum")
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, i+batch_size] = 0
            mask[i+batch_size, i] = 0
        self.register_buffer("mask", mask)

    def forward(self, z1, z2):
        N = 2*self.bs
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / self.temp
        pos = torch.cat([sim.diag(self.bs), sim.diag(-self.bs)], dim=0).view(N,1)
        neg = sim[self.mask].view(N, -1)
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.device)
        loss = self.crit(logits, labels)
        return loss / N

# main function for training
def main():
    # hyperparameters
    batch_size = 128
    lr         = 1e-4
    epochs     = 40
    temp       = 0.4
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zarr_path   = 'multimodal_training/replay_buffer.zarr'
    frames_root = 'multimodal_training/frames'
    transform = T.Compose([
        T.Resize((240,320)),
        T.RandomResizedCrop((240,320), scale=(0.9,1.0)),
        T.ColorJitter(0.2,0.2,0.2,0.05),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # dataset and loader
    ds = RobotArmDataset(zarr_path, frames_root, transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )

    #
    model   = FusionEncoder().to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = NTXentLoss(batch_size, temp, device).to(device)
    scaler  = torch.cuda.amp.GradScaler()

    # Main loop
    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        for v1, v2 in loader:
            opt.zero_grad()

            for d in (v1, v2):
                d['hand_img']  = d['hand_img'].to(device, non_blocking=True)
                d['third_img'] = d['third_img'].to(device, non_blocking=True)
                d['pose']      = d['pose'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                z1 = model(v1['hand_img'], v1['third_img'], v1['pose'])
                z2 = model(v2['hand_img'], v2['third_img'], v2['pose'])
                loss = loss_fn(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        print(f"Epoch {ep}/{epochs} — Avg Loss: {running/len(loader):.4f}")

    torch.save(model.state_dict(), "fusion_encoder_contrastive_fast.pth")

if __name__ == "__main__":
    main()

