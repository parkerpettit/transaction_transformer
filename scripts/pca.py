import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

# --- Load your saved embeddings and labels ---
X = np.load("data/X_train.npy")  # shape = [N, D]
y = np.load("data/Y_train.npy")  # shape = [N]
print("Done loading embeddings and labels")

# --- Balanced subsample: all fraud + equal non-fraud ---
rng = np.random.default_rng(42)
fraud_idx    = np.where(y == 1)[0]
nonfraud_idx = np.where(y == 0)[0]
n_fraud = len(fraud_idx)

# sample exactly n_fraud non-fraud points
nonfraud_sample = rng.choice(nonfraud_idx, size=n_fraud, replace=False)

# combine and shuffle
idx = np.concatenate([fraud_idx, nonfraud_sample])
rng.shuffle(idx)
X_sample = X[idx]
y_sample = y[idx]
print(f"Subsample size: {len(idx)} (fraud={n_fraud}, non-fraud={n_fraud})")

# --- Pre‑reduce dimensionality with PCA (optional but recommended) ---
print("Starting PCA reduction to 30 dims")
# pca = PCA(n_components=100, random_state=42)
# X_reduced = pca.fit_transform(X_sample)

# --- Run UMAP on the reduced data ---
print("Starting UMAP projection to 2D")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    verbose=True
    # random_state=42
)
X_umap = reducer.fit_transform(X_sample)
print("Done UMAP projection")

# --- Plot the 2D UMAP embedding ---
plt.figure(figsize=(8, 6))
palette = {0: '#1f77b4', 1: '#d62728'}  # blue = non-fraud, red = fraud

sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=y_sample,
    palette=palette,
    alpha=0.6,
    edgecolor='k',
    linewidth=0.2,
    s=20
)

plt.title("UMAP of Balanced Transformer Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Class", labels=["Non-Fraud", "Fraud"])
plt.grid(True)
plt.tight_layout()
plt.show()
