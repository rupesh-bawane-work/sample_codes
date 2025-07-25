import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df[:,:10]
true_labels = df[:,11]

# === 2. PCA for 2D visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# === 3. Agglomerative Clustering ===
n_clusters = 3
agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = agglo.fit_predict(X)

# === 4. Plot clusters in PCA-reduced space ===
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=50)
plt.title(f'Agglomerative Clustering (n_clusters={n_clusters}) - PCA 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# === 5. Plot dendrogram ===
plt.figure(figsize=(10, 5))
linked = linkage(X, method='ward')
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# === 6. Export results ===
final_df = X.copy()
final_df['Cluster'] = cluster_labels
final_df['PCA1'] = X_pca[:, 0]
final_df['PCA2'] = X_pca[:, 1]
final_df['True_Label'] = true_labels

final_df.to_csv('agglomerative_clustered_with_pca.csv', index=False)
