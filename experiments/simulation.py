# Class 0 (female-associated) vs Class 1 (male-associated)
female_words = ["she", "nurse", "woman", "girl", "mother"]
male_words = ["he", "doctor", "man", "boy", "father"]
neutral_words = ["person", "individual", "child", "human"]

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)
random.seed(42)

d = 32  # embedding dimension
num_samples = 100

# Define base embeddings for each word
def random_vector(base=None, offset=0.2):
    if base is None:
        return torch.randn(d)
    else:
        return base + offset * torch.randn(d)

# Define "true" sensitive direction (gender)
gender_dir = torch.randn(d)
gender_dir = gender_dir / gender_dir.norm()

# Class 0 = female-associated
X_female = torch.stack([random_vector(base=gender_dir) for _ in range(num_samples)])
y_female = torch.zeros(num_samples)

# Class 1 = male-associated
X_male = torch.stack([random_vector(base=-gender_dir) for _ in range(num_samples)])
y_male = torch.ones(num_samples)

# Combine
X = torch.cat([X_female, X_male], dim=0)
y = torch.cat([y_female, y_male], dim=0)

# sensitive direction
# Difference vectors
D = X_male - X_female
U, S, V = torch.pca_lowrank(D)
v_sensitive_est = V[:, 0]  # top-1 principal direction

# Cosine similarity 확인
cos_sim = torch.cosine_similarity(v_sensitive_est, gender_dir.unsqueeze(0), dim=-1)
print("추정된 민감 방향 vs ground-truth gender direction cosine similarity:", cos_sim.item())

from sklearn.decomposition import PCA

X_np = X.detach().numpy()
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_np)

plt.figure(figsize=(6, 4))
plt.scatter(X_2d[:num_samples, 0], X_2d[:num_samples, 1], label='female', alpha=0.6)
plt.scatter(X_2d[num_samples:, 0], X_2d[num_samples:, 1], label='male', alpha=0.6)
plt.legend()
plt.title("PCA Projection of Synthetic Gendered Embeddings")
plt.grid(True)
plt.show()



