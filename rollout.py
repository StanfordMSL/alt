import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import zarr
from torchvision.models import resnet18, ResNet18_Weights

# preprocessing
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])



# define the model - should be consistent with the training script
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
        # mode 1: hand_img + third_img + pose
        # self.pose_encoder = PoseEncoder(pose_embed)
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(img_embed * 2 + pose_embed, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, final_embed)
        # )
        # mode 2: hand_img + third_img
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(img_embed * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, final_embed)
        )
        # mode 3: third_img
        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(img_embed, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, final_embed)
        # )

    def forward(self, hand_img, third_img, pose):
        # hand_img, third_img: [B, 3, H, W]；pose: [B, 7]
        hand_feat = self.image_encoder(hand_img)  # [B, img_embed]
        third_feat = self.image_encoder(third_img)  # [B, img_embed]
        # pose_feat = self.pose_encoder(pose)  # [B, pose_embed]
        # fused = torch.cat([hand_feat, third_feat, pose_feat], dim=1)
        fused = torch.cat([hand_feat, third_feat], dim=1)
        # fused = third_feat
        embedding = self.fc(fused)
        # L2 normalization, make cosine similarity calculation more stable
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding


# loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionEncoder().to(device)
model.load_state_dict(torch.load("fusion_encoder_contrastive.pth", map_location=device))
model.eval()

# loading the trajectory database
db = torch.load("traj_database.pt", map_location=device)
db_embeddings = db['embeddings'].to(device)  # [N, final_embed]
info_list = db['info']

# loading the zarr file
zarr_path = "rgb_training/replay_buffer.zarr"
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


# define a safe position
def get_safety_position():
    """
    return a safe EEF position (7-dim)
    Currently, the safe_eef is not work for the robot arm
    """
    safe_eef = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return safe_eef


# real-time model inference
similarity_threshold = 0.5
t_steps = 30  # future steps to predict


def realtime_rollout(hand_img_cv, third_img_cv, pose_np):
    """
    main function for real-robot experiment
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
    similarity = torch.matmul(db_embeddings, emb.T).squeeze()  #
    max_sim, max_idx = torch.max(similarity, dim=0)

    # if the max similarity is lower than the threshold, return a safe position
    if max_sim.item() < similarity_threshold:
        print("OOD detected, return a safe position")
        return get_safety_position()

    # find the matched episode and local index
    matched_episode, matched_local_idx = info_list[max_idx]
    global_idx = get_global_index(matched_episode, matched_local_idx)
    print(
        f"matched episode: {matched_episode}, local frame: {matched_local_idx}, global frame: {global_idx}, sim: {max_sim.item():.3f}")

    # output the future trajectory
    future_traj = get_future_trajectory(global_idx, t_steps)
    return future_traj, matched_episode


def get_poses():
    # from zarr find the episode starts
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1], axis=0)  # (num_episode,)
    # get the low-dim data of the start frame of each episode
    start_poses = []
    for start_idx in episode_starts:
        pos = robot_eef_pos[start_idx]
        quat = robot_eef_quat[start_idx]
        pose = np.concatenate([pos, quat], axis=0)  # (7,)
        start_poses.append(pose)
    start_poses = np.stack(start_poses, axis=0)  # (num_episode, 7)
    return start_poses

# test
if __name__ == "__main__":
    import time
    pose_np = get_poses()

    ###############################
    # InD test
    ###############################
    # from rollout import realtime_rollout
    total_cases = 31  # 31 for InD, 8 for OoD
    success_count = 0
    failed_cases = []

    for idx in range(total_cases):
        hand_img_path = f"InD_cases/hand{idx}.jpg"  # InD -> OoD; jpg->png
        third_img_path = f"InD_cases/eye{idx}.jpg"  #

        hand_img_cv = cv2.imread(hand_img_path)
        third_img_cv = cv2.imread(third_img_path)

        time_start = time.time()
        future_trajectory, matched_episode = realtime_rollout(hand_img_cv, third_img_cv, pose_np[idx])
        time_end = time.time()
        # print(f"Time taken for test {idx}: {time_end - time_start:.8f} seconds")

        # print(f"Future trajectory: {future_trajectory}")

        if matched_episode == idx:  # idx for test_cases, new_cases should be 5
            print(f"[Test {idx}] Success! Matched Episode: {matched_episode}")
            success_count += 1
        else:
            print(f"[Test {idx}] Fail! Matched Episode: {matched_episode}")
            failed_cases.append((idx, matched_episode))

    success_rate = success_count / total_cases * 100
    print("\n========================================")
    print(f"Total Cases: {total_cases}")
    print(f"Successful Matches: {success_count}")
    print(f"Failed Matches: {total_cases - success_count}")
    print(f"Match Success Rate: {success_rate:.2f}%")
    print("========================================")

    if failed_cases:
        print("Failed cases details (test_index → matched_episode):")
        for test_idx, matched_ep in failed_cases:
            print(f"  Test {test_idx} → matched {matched_ep}")

    ###############################
    # Out-of-Distribution (OoD) test
    ###############################
    # from rollout import realtime_rollout
    total_cases = 8      # 31 for InD, 8 for OoD
    success_count = 0
    failed_cases = []

    for idx in range(total_cases):
        hand_img_path = f"OoD_cases/hand{idx}.png"   #  InD -> OoD; jpg->png
        third_img_path = f"OoD_cases/eye{idx}.png"   #

        hand_img_cv = cv2.imread(hand_img_path)
        third_img_cv = cv2.imread(third_img_path)

        time_start = time.time()
        future_trajectory, matched_episode = realtime_rollout(hand_img_cv, third_img_cv, pose_np[idx])
        time_end = time.time()
        # print(f"Time taken for test {idx}: {time_end - time_start:.8f} seconds")

        # print(f"Future trajectory: {future_trajectory}")

        if matched_episode == 5:  # idx for test_cases, new_cases should be 5
            print(f"[Test {idx}] Success! Matched Episode: {matched_episode}")
            success_count += 1
        else:
            print(f"[Test {idx}] Fail! Matched Episode: {matched_episode}")
            failed_cases.append((idx, matched_episode))

    success_rate = success_count / total_cases * 100
    print("\n========================================")
    print(f"Total Cases: {total_cases}")
    print(f"Successful Matches: {success_count}")
    print(f"Failed Matches: {total_cases - success_count}")
    print(f"Match Success Rate: {success_rate:.2f}%")
    print("========================================")

    if failed_cases:
        print("Failed cases details (test_index → matched_episode):")
        for test_idx, matched_ep in failed_cases:
            print(f"  Test {test_idx} → matched {matched_ep}")

