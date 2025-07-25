import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df[:,:10]
true_labels = df[:,11]

# === 2. Standardize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Apply PCA for 2D visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === 4. Apply DBSCAN clustering ===
dbscan = DBSCAN(eps=0.6, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

# Check number of clusters (excluding noise label -1)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print(f"✅ DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

# === 5. Visualize DBSCAN results (PCA-reduced) ===
plt.figure(figsize=(8, 5))
unique_labels = set(cluster_labels)
colors = [plt.cm.tab10(i) for i in range(len(unique_labels))]

for label in unique_labels:
    mask = cluster_labels == label
    label_name = f'Cluster {label}' if label != -1 else 'Noise'
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label_name, s=50, edgecolors='k')

plt.title('DBSCAN Clustering (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 6. Export results ===
final_df = X.copy()
final_df['Cluster'] = cluster_labels
final_df['PCA1'] = X_pca[:, 0]
final_df['PCA2'] = X_pca[:, 1]
final_df['True_Label'] = true_labels

final_df.to_csv('dbscan_clustered_with_pca.csv', index=False)
