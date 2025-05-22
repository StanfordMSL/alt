import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import zarr
from torchvision.models import resnet18, ResNet18_Weights


preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, embed_dim)

    def forward(self, x):
        return self.cnn(x)


# class PoseEncoder(nn.Module):
#     def __init__(self, embed_dim=32):
#         super().__init__()
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(7, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, embed_dim)
#         )
#
#     def forward(self, x):
#         return self.mlp(x)


class FusionEncoder(nn.Module):
    def __init__(self, img_embed=128, pose_embed=32, final_embed=128):
        super().__init__()
        self.image_encoder = ImageEncoder(img_embed)
        # self.pose_encoder = PoseEncoder(pose_embed)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(img_embed * 2 + pose_embed, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, final_embed)
        # )
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(img_embed * 2, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, final_embed)
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(img_embed, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, final_embed)
        )

    def forward(self, hand_img, third_img, pose):
        # hand_img, third_img: [B, 3, H, W]；pose: [B, 7]
        # hand_feat = self.image_encoder(hand_img)  # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        # pose_feat = self.pose_encoder(pose)  # [B, pose_embed]
        # fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        # fused = torch.cat([hand_feat, third_feat], dim=1)
        fused = third_feat
        embedding = self.fc(fused)
        # L2 normalization, make cosine similarity calculation more stable
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionEncoder().to(device)
model.load_state_dict(torch.load("fusion_encoder_contrastive_fast.pth", map_location=device))
model.eval()

# loading the trajectory database
db = torch.load("traj_database.pt", map_location=device)
db_embeddings = db['embeddings'].to(device)
info_list = db['info']


zarr_path = "multimodal_training/replay_buffer.zarr"
zarr_root = zarr.open(zarr_path, mode='r')
robot_eef_pos = zarr_root["data/robot_eef_pos"][:]
robot_eef_quat = zarr_root["data/robot_eef_quat"][:]
episode_ends = zarr_root["meta/episode_ends"][:]

# start and end frame indices for each episode
episode_boundaries = []
start = 0
for end in episode_ends:
    episode_boundaries.append((start, end))
    start = end


def get_global_index(episode, local_idx):
    start, end = episode_boundaries[episode]
    return start + local_idx


def get_future_trajectory(global_idx, t):

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



def get_safety_position():
    """
    return a safe EEF position (7-dim)
    TODO: we need to find a proper safe position
    """
    safe_eef = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return safe_eef



similarity_threshold = 0.5
t_steps = 20

def realtime_rollout(hand_img_cv, third_img_cv, pose_np):

    # 1. Preprocess inputs
    hand_rgb  = cv2.cvtColor(hand_img_cv,  cv2.COLOR_BGR2RGB)
    third_rgb = cv2.cvtColor(third_img_cv, cv2.COLOR_BGR2RGB)
    h_t = preprocess(hand_rgb).unsqueeze(0).to(device)
    t_t = preprocess(third_rgb).unsqueeze(0).to(device)
    p_t = torch.tensor(pose_np, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(h_t, t_t, p_t)  # [1, D]

    sim_all = torch.matmul(db_embeddings, emb.T).squeeze(1)  # [N]

    sorted_sims, sorted_idxs = torch.sort(sim_all, descending=True)

    unique_eps   = []
    unique_idxs  = []
    unique_sims  = []
    for sim_val, db_idx in zip(sorted_sims.cpu().tolist(),
                               sorted_idxs.cpu().tolist()):
        ep, _ = info_list[db_idx]
        if ep in unique_eps:
            continue
        unique_eps.append(ep)
        unique_idxs.append(db_idx)
        unique_sims.append(sim_val)
        if len(unique_eps) == 3:
            break

    if not unique_sims or unique_sims[0] < similarity_threshold:
        return [get_safety_position()]*3, [-1]*3, [0.0]*3

    future_trajs = []
    for db_idx in unique_idxs:
        ep, local = info_list[db_idx]
        gidx = get_global_index(ep, local)
        future_trajs.append(get_future_trajectory(gidx, t_steps))

    return future_trajs, unique_eps, unique_sims


def get_poses():
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1], axis=0)

    start_poses = []
    for start_idx in episode_starts:
        pos = robot_eef_pos[start_idx]
        quat = robot_eef_quat[start_idx]
        pose = np.concatenate([pos, quat], axis=0)
        start_poses.append(pose)
    start_poses = np.stack(start_poses, axis=0)  # (num_episode, 7)
    return start_poses


if __name__ == "__main__":
    import time
    pose_np = get_poses()

    total_cases = 31
    success_count = 0
    failed_cases = []

    for idx in range(total_cases):
        hand_img_path  = f"InD/hand{idx}.jpg"
        third_img_path = f"InD/eye{idx}.jpg"
        hand_cv  = cv2.imread(hand_img_path)
        third_cv = cv2.imread(third_img_path)

        t0 = time.time()
        future_trajs, matched_eps, sims = realtime_rollout(hand_cv, third_cv, pose_np[idx])
        t1 = time.time()

        print(f"\n[Test {idx}] Time: {t1-t0:.4f}s")

        for rank, (traj, ep, sim) in enumerate(zip(future_trajs, matched_eps, sims), start=1):
            print(f"  Match {rank}: Episode={ep}, Similarity={sim:.3f}")



