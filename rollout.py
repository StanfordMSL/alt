import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import zarr
from torchvision import models

##############################
# 1. preprocess input images
##############################
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


##############################
# 2. define the model - should be consistent with the training script
##############################
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, x):
        return self.cnn(x)


class PoseEncoder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(7, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FusionEncoder(nn.Module):
    def __init__(self, img_embed=128, pose_embed=32, final_embed=128):
        super().__init__()
        self.image_encoder = ImageEncoder(img_embed)
        self.pose_encoder = PoseEncoder(pose_embed)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(img_embed * 2 + pose_embed, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, final_embed)
        )

    def forward(self, hand_img, third_img, pose):
        # hand_img, third_img: [B, 3, H, W]；pose: [B, 7]
        hand_feat = self.image_encoder(hand_img)  # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        pose_feat = self.pose_encoder(pose)  # [B, pose_embed]
        fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        embedding = self.fc(fused)
        # L2 normalization, make cosine similarity calculation more stable
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding


##############################
# 3. load the trained model and database
##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionEncoder().to(device)
model.load_state_dict(torch.load("fusion_encoder_contrastive.pth", map_location=device))
model.eval()

# loading the trajectory database
db = torch.load("traj_database.pt", map_location=device)
db_embeddings = db['embeddings'].to(device)  # [N, final_embed]
info_list = db['info']

##############################
# 4. load zarr file, get access the low-dim data and episode boundaries
##############################
zarr_path = "rgb_training/replay_buffer.zarr"  # 请根据实际路径修改
zarr_root = zarr.open(zarr_path, mode='r')
robot_eef_pos = zarr_root["data/robot_eef_pos"][:]  # (N, 3)
robot_eef_quat = zarr_root["data/robot_eef_quat"][:]  # (N, 4)
episode_ends = zarr_root["meta/episode_ends"][:]  # (num_episode,)

# start and end frame indices for each episode
episode_boundaries = []
start = 0
for end in episode_ends:
    episode_boundaries.append((start, end))
    start = end


def get_global_index(episode, local_idx):
    """
    according to the episode and local index, get the global index
    """
    start, end = episode_boundaries[episode]
    return start + local_idx


def get_future_trajectory(global_idx, t):
    """
    from the global index, get the future t steps trajectory
    each step is 7-dim (EEF: position + quaternion)
    if the future steps are not enough, pad the last step
    return a tensor of shape [t, 7]
    """
    total_frames = robot_eef_pos.shape[0]
    future = []
    for idx in range(global_idx, min(global_idx + t, total_frames)):
        pos = robot_eef_pos[idx]  # (3,)
        quat = robot_eef_quat[idx]  # (4,)
        frame_vec = torch.tensor(np.concatenate([pos, quat], axis=0), dtype=torch.float)
        future.append(frame_vec)
    if len(future) < t:
        pad = [future[-1]] * (t - len(future))
        future.extend(pad)
    return torch.stack(future)


##############################
# 5. define a safety position
##############################
def get_safety_position():
    """
    return a safe EEF position (7-dim)
    TODO: we need to find a proper safe position
    """
    safe_eef = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return safe_eef


##############################
# 6. the main function for real-time rollout
##############################
# set the similarity threshold
similarity_threshold = 0.5
# output future t_steps (50 in practice)
t_steps = 3


def realtime_rollout(hand_img_cv, third_img_cv, pose_np):
    """
    INPUT：
      hand_img_cv, third_img_cv：the real-time input images (BGR format)
      pose_np： the real-time pose data (7-dim EEF data)
    OUTPUT：
      if match successfully, return the future t steps trajectory (tensor, shape [t, 7])
      if match failed, return a safe position (7-dim)
    """
    # bgr -> rgb
    hand_img_rgb = cv2.cvtColor(hand_img_cv, cv2.COLOR_BGR2RGB)
    third_img_rgb = cv2.cvtColor(third_img_cv, cv2.COLOR_BGR2RGB)
    hand_tensor = preprocess(hand_img_rgb).unsqueeze(0).to(device)
    third_tensor = preprocess(third_img_rgb).unsqueeze(0).to(device)
    # convert np pose to tensor, add batch dim
    pose_tensor = torch.tensor(pose_np, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(hand_tensor, third_tensor, pose_tensor)  # [1, final_embed]

    # calculate cosine similarity with database embeddings
    similarity = torch.matmul(db_embeddings, emb.T).squeeze()  # [N]
    max_sim, max_idx = torch.max(similarity, dim=0)

    # if the max similarity is lower than the threshold, return a safe position
    if max_sim.item() < similarity_threshold:
        print("匹配相似度过低，认为输入为 OOD，返回安全位置")
        return get_safety_position()

    # 从 info_list 中获取匹配的 (episode, local_frame_idx)
    matched_episode, matched_local_idx = info_list[max_idx]
    global_idx = get_global_index(matched_episode, matched_local_idx)
    print(
        f"匹配到 episode: {matched_episode}, local frame: {matched_local_idx}, 全局索引: {global_idx}, 相似度: {max_sim.item():.3f}")

    # 提取未来 t 步轨迹（7 维 EEF 数据）
    future_traj = get_future_trajectory(global_idx, t_steps)
    return future_traj


##############################
# 7. call the function
######################################
if __name__ == "__main__":
    # real-time input image
    hand_img_cv = cv2.imread("test_cases/hand30.jpg")
    third_img_cv = cv2.imread("test_cases/eye30.jpg")
    # suppose we have the real time pose data
    pose_np = np.array([259.093719, 2.891501, 258.123199, 1.05087541e-04, 9.99983194e-01, -4.12921014e-03, -4.06809698e-03])
    # pose_np = np.array([0, 0, 0, 0, 0, 0, 0])

    future_trajectory = realtime_rollout(hand_img_cv, third_img_cv, pose_np)
    print("The selected trajectory is：")
    print(future_trajectory)
