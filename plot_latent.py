import torch

device = torch.device("cpu")

db = torch.load("traj_database.pt", map_location=device)
db_embeddings = db['embeddings'].to(device)  # [N, final_embed]
info_list = db['info']

print("Loaded database with {} embeddings.".format(len(info_list)))
print("Each embedding has shape: {}".format(db_embeddings.shape))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

emb_np = db_embeddings.cpu().numpy()
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_np)
x, y = emb_2d[:, 0], emb_2d[:, 1]

angles = (np.arctan2(y, x) + np.pi) / (2 * np.pi)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=angles, cmap='hsv', s=1100, marker='.', alpha=0.8)
plt.axis('off')
plt.tight_layout()
plt.show()


