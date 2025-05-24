import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from matplotlib.collections import LineCollection
import multiprocessing
multiprocessing.set_start_method('spawn', force=True) 
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
import sys  
import os
import imageio
from matplotlib.colors import to_rgb

from scipy.interpolate import splprep, splev
import shutil



# Configuration
# shape_type = 'star'    
shape_size = 2.0 # radius
noise_scale_factor = 0.5 * shape_size # for both training and inference      
# noise_scale = shape_size
plot_limit = shape_size * 1.5

# Low-data and high-data configurations
low_data_samples = 20      # Few training samples
high_data_samples = 100000    # Many training samples
num_epochs_small_data = 1000
num_epochs_big_data = 100
small_data_repeat_factor = 1000 # same as increasing the number of epochs

save_flow_video = True

# # SMALL TEST CASE
# low_data_samples = 20      # Few training samples
# high_data_samples = 100    # Many training samples
# num_epochs_small_data = 100
# num_epochs_big_data = 10


print_num_params = False

num_inference_samples = 5000
early_stopping_patience = num_epochs_small_data

use_clustering = False
num_clusters = 100
num_timesteps = 100         
hidden_size_simple = 24      
hidden_layers_simple = 1    
hidden_size_complex = 1024   
hidden_layers_complex = 10    
learning_rate_normal = 1e-3
learning_rate_complex_small_data = 1e-4
batch_size_low = 4096 #1024
batch_size_high = 4096 #16384             
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reset seed for reproducibility
np.random.seed(42)

def cluster_trajectories(trajectories, num_clusters=100):

    final_points = trajectories[:, -1, :]  # (N, 2)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(final_points)

    # Find one representative trajectory for each cluster (closest one to centroid)
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []
    for c in range(num_clusters):
        cluster_mask = (labels == c)
        cluster_members = final_points[cluster_mask]
        cluster_trajs = trajectories[cluster_mask]
        
        # remove small clusters
        if len(cluster_members) < 10:
            continue

        if len(cluster_members) == 0:
            continue

        dists = np.linalg.norm(cluster_members - cluster_centers[c], axis=1)
        best_idx = np.argmin(dists)
        selected_indices.append(cluster_mask.nonzero()[0][best_idx])

    selected_trajectories = trajectories[selected_indices]
    return selected_trajectories

# def sample_near_vertex_edges(vertices, n_samples, vertex_keep_prob=1.0, min_spread=0.1, max_spread=0.1):
#     """
#     Sample points near selected vertices along both sides (connected edges), no noise.
    
#     - Randomly drop some vertices (keep probability).
#     - Use different spread distances for different vertices.
#     """
#     n_vertices = len(vertices)
    
#     # Randomly decide which vertices are kept
#     keep_mask = np.random.rand(n_vertices) < vertex_keep_prob
#     active_vertices = np.where(keep_mask)[0]
    
#     if len(active_vertices) == 0:
#         raise ValueError("No active vertices! Try increasing vertex_keep_prob.")
    
#     # Assign random spread to each active vertex
#     spreads = {v: np.random.uniform(min_spread, max_spread) for v in active_vertices}

#     samples = []
    
#     for _ in range(n_samples):
#         vidx = np.random.choice(active_vertices)  # pick an active vertex
#         p0 = vertices[vidx]
#         p_prev = vertices[(vidx - 1) % n_vertices]
#         p_next = vertices[(vidx + 1) % n_vertices]
        
#         # Randomly choose one of the two connected edges
#         p1 = p_prev if np.random.rand() < 0.5 else p_next
        
#         # Sample close to vertex based on its spread
#         spread = spreads[vidx]
#         t = np.random.uniform(0.0, spread)
#         point = (1 - t) * p0 + t * p1
#         samples.append(point)
    
#     return np.stack(samples, axis=0)

def ellipse_func(t, size):
    long_axis = size  # x-axis stretched 2x
    short_axis = size * 0.5     # y-axis normal
    x = long_axis * np.cos(t)
    y = short_axis * np.sin(t)
    return x, y

def heart_func(t, size):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    x *= size / 20
    y *= size / 20
    return x, y

def sample_segments_along_curve(curve_func, size, n_segments, n_samples, segment_fraction=0.03, keep_prob=0.7):
    total_angle = 2 * np.pi
    segment_length = segment_fraction * total_angle

    samples = []

    # Evenly spaced starting points for each segment
    t_starts = np.linspace(0, total_angle, n_segments, endpoint=False)

    # Randomly decide which segments to keep
    keep_mask = np.random.rand(n_segments) < keep_prob
    kept_t_starts = t_starts[keep_mask]
    
    if len(kept_t_starts) == 0:
        raise ValueError("No segments kept! Try increasing keep_prob.")

    samples_per_segment = n_samples // len(kept_t_starts)

    for t0 in kept_t_starts:
        t1 = t0 + segment_length
        t = np.random.uniform(t0, t1, size=(samples_per_segment,))
        x, y = curve_func(t, size)
        pts = np.stack([x, y], axis=1)
        samples.append(pts)

    samples = np.concatenate(samples, axis=0)

    # Clean Resampling: Duplicate if needed
    if samples.shape[0] < n_samples:
        print(f"Warning: {samples.shape[0]} samples generated, but {n_samples} required. Resampling...")
        indices = np.random.choice(samples.shape[0], size=n_samples, replace=True)
        samples = samples[indices]
    else:
        samples = samples[:n_samples]

    return samples

def sample_along_edges_uniformly(vertices, n_samples):
    edges = [(vertices[i], vertices[(i+1) % len(vertices)]) for i in range(len(vertices))]
    lengths = [np.linalg.norm(b - a) for a, b in edges]
    total_length = sum(lengths)
    distances = np.linspace(0, total_length, n_samples, endpoint=False)

    samples = []
    acc = 0
    i = 0
    for d in distances:
        while d > acc + lengths[i]:
            acc += lengths[i]
            i += 1
        a, b = edges[i]
        t = (d - acc) / lengths[i]
        samples.append((1 - t) * a + t * b)
    return np.stack(samples)

def sample_shape_points(shape: str, size: float, n_samples: int, near_vertex_edges=False):
    if shape == 'ellipse':
        if near_vertex_edges:
            data = sample_segments_along_curve(ellipse_func, size, n_segments=20, n_samples=n_samples, segment_fraction=0.01, keep_prob=0.5)
        else:
            angles = np.random.rand(n_samples) * 2 * np.pi
            long_axis = size 
            short_axis = size * 0.5
            x = long_axis * np.cos(angles)
            y = short_axis * np.sin(angles)
            data = np.stack([x, y], axis=1)


    elif shape == 'star':
        R = size
        r = 0.5 * size
        outer_angles = np.linspace(np.pi/2, 5*np.pi/2, 6)[:-1]
        inner_angles = outer_angles + np.pi/5  
        outer_pts = np.stack([R * np.cos(outer_angles), R * np.sin(outer_angles)], axis=1)
        inner_pts = np.stack([r * np.cos(inner_angles), r * np.sin(inner_angles)], axis=1)
        vertices = []
        for i in range(5):
            vertices.append(outer_pts[i])
            vertices.append(inner_pts[i])
        vertices = np.array(vertices)

        if near_vertex_edges:
            data = sample_near_vertex_edges(vertices, n_samples, min_spread=0.2, max_spread=0.2, vertex_keep_prob=0.7)
            # convex_vertices = vertices[::2]  # take every second point: 0, 2, 4, 6, 8
            # data = sample_near_vertex_edges(convex_vertices, n_samples, size)
        else:
            data = sample_along_edges_uniformly(vertices, n_samples)

    elif shape == 'heart':
        if near_vertex_edges:
            data = sample_segments_along_curve(heart_func, size, n_segments=25, n_samples=n_samples, segment_fraction=0.02, keep_prob=0.5)
        else:
            # Full random t for smooth heart
            t = np.random.rand(n_samples) * 2 * np.pi
            x = 16 * np.sin(t)**3
            y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            x *= size / 20  # scaling down
            y *= size / 20
            data = np.stack([x, y], axis=1)

    elif shape == 'rectangle':
        # Rectangle with width and height = 2 * size
        half_w = size
        half_h = size
        vertices = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ], dtype=np.float32)

        if near_vertex_edges:
            # Small data: sample near 4 corners
            data = sample_near_vertex_edges(vertices, n_samples, max_spread=0.1)
        else:
            # Big data: sample uniformly along rectangle edges
            data = sample_along_edges_uniformly(vertices, n_samples)

    else:
        raise ValueError(f"Unknown shape_type: {shape}")
        
    
    return data.astype(np.float32)
# points_np = sample_shape_points(shape_type, shape_size, num_training_points)
# points = torch.from_numpy(points_np).to(device)  

# Model
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 2, hidden_size: int = 128, hidden_layers: int = 3):
        super(DiffusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Create simple and complex models
simple_model = DiffusionMLP(input_dim=3, output_dim=2, hidden_size=hidden_size_simple, hidden_layers=hidden_layers_simple).to(device)
complex_model = DiffusionMLP(input_dim=3, output_dim=2, hidden_size=hidden_size_complex, hidden_layers=hidden_layers_complex).to(device)

# calculate and print the number of parameters
if print_num_params:
    num_params_simple = sum(p.numel() for p in simple_model.parameters())
    num_params_complex = sum(p.numel() for p in complex_model.parameters())
    print(f"Simple model parameters: {num_params_simple: .2f}")
    print(f"Complex model parameters: {num_params_complex / 1e6:.2f}M")

# hyperparameters
T = num_timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)           
alphas = 1 - betas                                        
alpha_bars = torch.cumprod(alphas, dim=0)                 

# Training
# with early stopping and best model restoration
def train_model(model, points, num_epochs, learning_rate, label=None, patience=early_stopping_patience, batch_size=None, shape_type=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    loss_fn = nn.MSELoss()
    print(f"[{shape_type.upper()}] -> Starting training: {label}")
    model.train()
    epoch_losses = []

    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(points.shape[0])
        points_shuffled = points[perm]
        running_loss = 0.0

        for i in range(0, points_shuffled.size(0), batch_size):
            x0 = points_shuffled[i:i+batch_size]
            t = torch.randint(low=1, high=T+1, size=(x0.size(0),), device=device)
            alpha_bar_t = alpha_bars[t-1].unsqueeze(1)
            epsilon = noise_scale_factor * torch.randn_like(x0)
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
            t_norm = (t - 1).unsqueeze(1).float() / (T - 1)
            model_input = torch.cat([x_t, t_norm], dim=1)
            pred_epsilon = model(model_input)
            loss = loss_fn(pred_epsilon, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x0.size(0)

        scheduler.step()  # Update learning rate

        epoch_loss = running_loss / points.shape[0]
        epoch_losses.append(epoch_loss)

        # Optional: log learning rate
        current_lr = scheduler.get_last_lr()[0]
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Shape: {shape_type}]: Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[Shape: {shape_type}]: Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Shape: {shape_type}]: Restored best model with loss {best_loss:.6f}")
    else:
        print("Error: [Shape: {shape_type}]: No best model state found. Training may not have been successful.")
        sys.exit(1)

    return epoch_losses


# sampling
def sample_points_with_trajectory(model, num_samples, initial_noise):
    x_t = initial_noise.to(device)
    trajectories = [x_t.cpu().numpy()]
    for t in range(T, 0, -1):
        t_cur = torch.full((num_samples,), t, device=device, dtype=torch.long)
        t_norm = (t_cur - 1).unsqueeze(1).float() / (T - 1)
        model_input = torch.cat([x_t, t_norm], dim=1)
        with torch.no_grad():
            pred_epsilon = model(model_input)
        alpha_bar_t = alpha_bars[t-1]
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_epsilon) / torch.sqrt(alpha_bar_t)
        if t > 1:
            z = torch.randn_like(x_t)
            alpha_bar_prev = alpha_bars[t-2] if t-2 >= 0 else torch.tensor(1.0, device=device)
            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * z
        else:
            x_t = pred_x0
        trajectories.append(x_t.cpu().numpy())
    trajectories = np.stack(trajectories, axis=1)  # (num_samples, num_steps+1, 2)

    if use_clustering:
        clustered_trajectories = cluster_trajectories(trajectories, num_clusters=100)
    else:
        clustered_trajectories = trajectories
    return clustered_trajectories

def plot_trajectories(trajectories, points_np, title=None, last_steps=30, ax=None, plot_denoising_lines=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    num_trajectories, num_steps, _ = trajectories.shape
    steps_to_plot = min(last_steps, num_steps)


    if plot_denoising_lines:

        # Plot fading lines for the denoising steps
        # only plot 1000 trajectories, sampled evenly
        step_size = max(1, num_trajectories // 1000)
        for i in range(0, num_trajectories, step_size):
            
            traj = trajectories[i, -steps_to_plot::2, :]  # take last steps_to_plot steps, every 5th step
            x, y = traj[:, 0], traj[:, 1]

            try:
                # Fit a spline through the points
                tck, u = splprep([x, y], s=1.0, k=2)  # smoothing factor s, degree k
                u_fine = np.linspace(0, 1, 100)  # more points for smoothness
                x_fine, y_fine = splev(u_fine, tck)
                ax.plot(x_fine, y_fine, color='cyan', alpha=0.2, linewidth=0.2, zorder=1)
            except Exception as e:
                # fallback if spline fitting fails
                ax.plot(x, y, color='cyan', alpha=0.15, linewidth=0.2, zorder=1)
            
        
    
    # Plot
    start_points = trajectories[:, 0, :]
    end_points = trajectories[:, -1, :]
    connect_segments = np.stack([start_points, end_points], axis=1)
    # lc_connect = LineCollection(connect_segments, colors='green', linewidths=0.15, alpha=0.8, zorder=1)
    lc_connect = LineCollection(connect_segments, colors='cyan', linewidths=0.15, alpha=0.8, zorder=1)
    ax.add_collection(lc_connect)

    # init noise points
    ax.scatter(start_points[:, 0], start_points[:, 1],
               color='gray', edgecolors='gray', s=1, alpha=0.2, label="Input Points", zorder=3)

    # after denoising
    ax.scatter(end_points[:, 0], end_points[:, 1], 
               color='blue', s=40, edgecolors='white', linewidths=0.5, zorder=10, alpha=0.25)

    # gound truth points
    marker_size_cur = 30 if points_np.shape[0] < high_data_samples else 7
    ax.scatter(points_np[:, 0], points_np[:, 1], 
               c='orange', s=marker_size_cur, linewidths=0.1, alpha=1.0, zorder=100, label="Ground Truth")

    if title is not None:
        ax.set_title(title, fontsize=16)
        ax.legend(loc='upper right', fontsize=12)

    ax.axis('equal')
    ax.axis('off')

    # Fix the plot plot_limits based on shape size
    
    
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    
# Before training
all_losses = []
all_labels = []


def train_and_plot_all(shape_type):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    save_dir = f"{shape_type}"
    
    if os.path.exists(save_dir):
        print(f"[{shape_type.upper()}] Removing old folder: '{save_dir}'")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_losses = []
    all_labels = []

    points_np_low = sample_shape_points(shape_type, shape_size, low_data_samples, near_vertex_edges=False)
    points_np_high = sample_shape_points(shape_type, shape_size, high_data_samples, near_vertex_edges=False)

    points_low = torch.from_numpy(points_np_low).to(device)
    points_high = torch.from_numpy(points_np_high).to(device)

    configs = [
        (points_low, "Small Data + Simple Model", hidden_size_simple, hidden_layers_simple, axs[0], batch_size_low, "small"),
        (points_high, "Big Data + Simple Model", hidden_size_simple, hidden_layers_simple, axs[1], batch_size_high, "big"),
        (points_low, "Small Data + Complex Model", hidden_size_complex, hidden_layers_complex, axs[2], batch_size_low, "small"),
        (points_high, "Big Data + Complex Model", hidden_size_complex, hidden_layers_complex, axs[3], batch_size_high, "big")
    ]

    shared_initial_noise = noise_scale_factor * torch.randn(num_inference_samples, 2).to(device)
    title_names = []

    for idx, (points_tensor, label, hidden_size, hidden_layers, ax, batch_size, data_type) in enumerate(configs):
        if data_type == "small":
            points_tensor = points_tensor.repeat((small_data_repeat_factor, 1))
            num_epochs = num_epochs_small_data
            learning_rate = learning_rate_complex_small_data if hidden_size == hidden_size_complex else learning_rate_normal
        elif data_type == "big":
            num_epochs = num_epochs_big_data
            learning_rate = learning_rate_normal
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        model = DiffusionMLP(input_dim=3, output_dim=2, hidden_size=hidden_size, hidden_layers=hidden_layers).to(device)

        losses = train_model(
            model, points_tensor, num_epochs, learning_rate,
            label=label, patience=early_stopping_patience,
            batch_size=batch_size, shape_type=shape_type
        )
        all_losses.append(losses)
        all_labels.append(label)

        traj = sample_points_with_trajectory(model, num_samples=num_inference_samples, initial_noise=shared_initial_noise)
        points_np = points_tensor.cpu().numpy()
        
        if data_type == "small":
            data_size = low_data_samples
            title_name = f"{label} ({data_size} demos, {hidden_size}x{hidden_layers})"
        else:
            data_size = high_data_samples
            title_name = f"{label} ({data_size // 1000}k demos, {hidden_size}x{hidden_layers})"
        title_names.append(title_name)

        plot_trajectories(traj, points_np, title=title_name, last_steps=30, ax=ax)
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)

        fig_single, ax_single = plt.subplots(figsize=(6, 6))
        plot_trajectories(traj, points_np, title=None, last_steps=30, ax=ax_single)
        ax_single.set_xlim(-plot_limit, plot_limit)
        ax_single.set_ylim(-plot_limit, plot_limit)
        ax_single.axis('equal')
        ax_single.axis('off')

        model_type = "Simple" if hidden_size == hidden_size_simple else "Complex"
        cur_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cur_time_str += f"_{shape_type}_{shape_size}_layer_size_{hidden_size_simple}_{hidden_layers_simple}_{hidden_size_complex}_{hidden_layers_complex}_{num_epochs}epochs_data_size_{low_data_samples}_{high_data_samples}"
        save_name_single = f"shape_{shape_type}_Model_is_{model_type}_Data_is_{data_type.capitalize()}_{cur_time_str}.png"
        fig_single.savefig(os.path.join(save_dir, save_name_single), dpi=300)
        plt.close(fig_single)
        print(f"Saved subplot {idx+1} to '{save_name_single}'.")

        if save_flow_video:
            video_filename = os.path.join(save_dir, f"video_{shape_type}_{label.replace(' ', '_')}.mp4")
            save_trajectory_video(
                traj,
                shape_type=shape_type,
                save_path=video_filename,
                ground_truth_points=points_np,
            )

    plt.tight_layout()
    cur_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    cur_time_str += f"_{shape_type}_{shape_size}_layer_size_{hidden_size_simple}_{hidden_layers_simple}_{hidden_size_complex}_{hidden_layers_complex}_{num_epochs}epochs_data_size_{low_data_samples}_{high_data_samples}"
    save_name_full = f"trajectories_grid_{cur_time_str}.png"
    fig.savefig(save_name_full, dpi=300)
    print(f"Saved full grid figure to '{save_name_full}'.")
    plt.close(fig)

    plt.figure(figsize=(8, 6))
    for losses, label in zip(all_losses, all_labels):
        plt.plot(losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)

    save_name_loss = f"loss_curves_{cur_time_str}.png"
    plt.savefig(os.path.join(save_dir, save_name_loss), dpi=300)
    plt.close()
    print(f"Saved loss curves figure to '{save_name_loss}'.")

def train_and_plot_shape(shape):
    print(f"\n========================\nTraining on shape: {shape}\n========================")
    train_and_plot_all(shape)


def interpolate_color(c1, c2, alpha):
    return tuple((1 - alpha) * a + alpha * b for a, b in zip(c1, c2))

def save_trajectory_video(trajectories, shape_type, save_path="diffusion.mp4", fps=15, tail_length=20, ground_truth_points=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.collections import LineCollection
    import imageio

    num_samples, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []

    gray_rgb = to_rgb("gray")
    blue_rgb = to_rgb("blue")
    trail_color = (0.3, 0.3, 0.3)  
    min_size = 1
    max_size = 40
    subset = min(num_samples, 300)
    marker_size_cur = 30 if ground_truth_points.shape[0] < high_data_samples else 7


    last_rendered_frame = None

    for t in range(1, num_steps):
        ax.clear()
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.axis('off')
        ax.set_title(f"{shape_type.upper()} Diffusion Step {t}/{num_steps - 1}", fontsize=14)

        # trail
        segments = []
        segment_colors = []
        segment_widths = []

        overall_alpha_scale = t / (num_steps - 1)

        for i in range(subset):
            for dt in range(tail_length):
                t1 = t - dt - 1
                t2 = t1 + 1
                if t1 < 0 or t2 >= num_steps:
                    continue
                p1 = trajectories[i, t1]
                p2 = trajectories[i, t2]
                fade = dt / tail_length

                base_alpha = 0.1 + 0.2 * (1 - fade)
                alpha = base_alpha * overall_alpha_scale
                width = 0.6 + 0.5 * (1 - fade)

                segments.append([p1, p2])
                segment_colors.append((*trail_color, alpha))
                segment_widths.append(width)

        if segments:
            lc = LineCollection(
                segments,
                colors=segment_colors,
                linewidths=segment_widths,
                zorder=1
            )
            ax.add_collection(lc)

        # points
        alpha = t / (num_steps - 1)
        sizes = min_size + alpha * (max_size - min_size)
        colors = [interpolate_color(gray_rgb, blue_rgb, alpha) for _ in range(num_samples)]

        point_alpha = 0.25  # consistent with plot
        ax.scatter(
            trajectories[:, t, 0],
            trajectories[:, t, 1],
            s=sizes,
            c=colors,
            edgecolors='white',
            linewidths=0.2,
            alpha=point_alpha,
            zorder=2
        )
        # real
        if ground_truth_points is not None:
            ax.scatter(
                ground_truth_points[:, 0],
                ground_truth_points[:, 1],
                c='orange',
                s=marker_size_cur,
                linewidths=0.1,
                alpha=1.0,
                zorder=100
            )

        # save frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        last_rendered_frame = frame  # save last frame

    # Repeat the last frame 10 times to create a pause
    frames.extend([last_rendered_frame] * 10)

    imageio.mimsave(save_path, frames, fps=fps)
    plt.close(fig)
    print(f"Saved diffusion video to: {save_path}")

    
if __name__ == "__main__":
    shapes = ['star', 'ellipse', 'heart', 'rectangle']
    processes = []

    for shape in shapes:
        p = multiprocessing.Process(target=train_and_plot_shape, args=(shape,))
        p.start()
        processes.append(p)

    for p, shape in zip(processes, shapes):
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.pid} for shape '{shape}' exited with error code {p.exitcode}. Stopping program.")
            sys.exit(1)  

    print("All shapes finished successfully.")