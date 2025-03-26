import os
import cv2
import numpy as np
import zarr
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
import torch.nn as nn

########################################
# 1. 数据预处理：统一调整为 (240,320)
########################################
# 注意：PIL中尺寸以 (height, width) 表示，这里统一调整为 240x320
data_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),  # 将图像调整为 320x240 分辨率（即高度240，宽度320）
    # T.RandomHorizontalFlip(),
    # T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


########################################
# 2. 数据集类：RobotArmDataset
########################################
class RobotArmDataset(Dataset):
    def __init__(self, zarr_path, videos_dir, transform):
        """
        zarr_path: zarr 文件路径
        videos_dir: 存放各 episode 视频的文件夹路径（内部有 0,1,2,... 子文件夹，每个子文件夹内包含 '1.mp4' (hand view) 和 '3.mp4' (third view)）
        transform: 图像预处理及数据增强流水线
        """
        self.transform = transform
        self.videos_dir = videos_dir

        # 加载 zarr 文件
        zarr_root = zarr.open(zarr_path, mode='r')
        self.robot_eef_pos = zarr_root['data/robot_eef_pos'][:]  # (N, 3)
        self.robot_eef_quat = zarr_root['data/robot_eef_quat'][:]  # (N, 4)
        # 此处仅使用 EEF 数据（位置 + 四元数，共7维）
        self.episode_ends = zarr_root['meta/episode_ends'][:]  # (num_episode,)

        # 根据 episode_ends 计算每个 episode 的全局帧范围
        self.episode_boundaries = []
        start = 0
        for end in self.episode_ends:
            self.episode_boundaries.append((start, end))
            start = end
        self.total_frames = self.robot_eef_pos.shape[0]

        # 预先加载各 episode 视频帧（若数据量较大，可考虑按需加载）
        self.video_frames = {}  # 格式：{ episode_index: {'hand': [...], 'third': [...] } }
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
        # 根据全局索引确定当前帧所属 episode 以及在 episode 内的帧索引
        ep = 0
        local_idx = 0
        for i, (start, end) in enumerate(self.episode_boundaries):
            if idx < end:
                ep = i
                local_idx = idx - start
                break

        # 获取对应 episode 中的图像帧
        hand_img = self.video_frames[ep]['hand'][local_idx]  # numpy array, H x W x 3
        third_img = self.video_frames[ep]['third'][local_idx]

        # 从 zarr 中获取低维 EEF 数据（位置和四元数，合并为7维向量）
        pos = self.robot_eef_pos[idx]  # (3,)
        quat = self.robot_eef_quat[idx]  # (4,)
        pose = np.concatenate([pos, quat], axis=0)  # (7,)

        # 为对比学习生成两份不同的增强版本（这里只取 view1 用于构建数据库）
        hand_img_v1 = self.transform(hand_img.copy())
        third_img_v1 = self.transform(third_img.copy())
        # 对低维数据不做噪声添加（构建数据库时保持稳定性）
        pose_tensor = torch.tensor(pose, dtype=torch.float)

        view1 = {'hand_img': hand_img_v1, 'third_img': third_img_v1, 'pose': pose_tensor}
        # 这里返回 view1 两次，以保持与训练时 __getitem__ 的输出一致，但后续只使用第一个视图
        return view1, view1


########################################
# 3. 模型定义
########################################
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
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
        self.pose_encoder = PoseEncoder(pose_embed)
        self.fc = nn.Sequential(
            nn.Linear(img_embed * 2 + pose_embed, 256),
            nn.ReLU(),
            nn.Linear(256, final_embed)
        )

    def forward(self, hand_img, third_img, pose):
        # hand_img, third_img: [B, 3, H, W]，pose: [B, 7]
        hand_feat = self.image_encoder(hand_img)  # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        pose_feat = self.pose_encoder(pose)  # [B, pose_embed]
        fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        embedding = self.fc(fused)
        # L2归一化，使得余弦相似度计算更稳定
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding


########################################
# 4. 构建 traj_database.pt 脚本
########################################
if __name__ == "__main__":
    # 参数设置（请根据实际文件路径修改）
    zarr_path = 'rgb_training/replay_buffer.zarr'
    videos_dir = 'rgb_training/videos'
    save_path = 'traj_database.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型权重（确保 FusionEncoder 结构与训练时一致）
    model = FusionEncoder().to(device)
    model.load_state_dict(torch.load("fusion_encoder_contrastive.pth", map_location=device))
    model.eval()

    # 加载数据集（此数据集返回两份视图，但我们只取第一个 view）
    dataset = RobotArmDataset(zarr_path=zarr_path, videos_dir=videos_dir, transform=data_transforms)

    all_embeddings = []
    info_list = []

    # 遍历整个数据集，计算每帧 embedding 并记录信息
    for idx in range(len(dataset)):
        view1, _ = dataset[idx]  # 只使用 view1
        # view1 中包含 'hand_img', 'third_img', 'pose'
        hand_img = view1['hand_img'].unsqueeze(0).to(device)  # 增加 batch 维度
        third_img = view1['third_img'].unsqueeze(0).to(device)
        pose = view1['pose'].unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(hand_img, third_img, pose)  # 得到 [1, final_embed]
            emb = emb.squeeze(0).cpu()  # 变为 [final_embed]
            all_embeddings.append(emb)

        # 根据 dataset 内的 episode_boundaries 确定当前全局 idx 对应的 (episode, local_frame_idx)
        for ep_idx, (start, end) in enumerate(dataset.episode_boundaries):
            if idx < end:
                local_idx = idx - start
                info_list.append((ep_idx, local_idx))
                break

        if (idx + 1) % 100 == 0:
            print(f"处理 {idx + 1} / {len(dataset)} 帧")

    # 将所有 embedding 堆叠为一个 tensor，形状为 [N, final_embed]
    embeddings_tensor = torch.stack(all_embeddings)

    # 构造数据库字典，并保存到文件
    traj_database = {'embeddings': embeddings_tensor, 'info': info_list}
    torch.save(traj_database, save_path)
    print(f"traj_database 已保存至 {save_path}")
