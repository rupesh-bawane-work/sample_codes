import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df.iloc[:,:10]
true_labels = df.iloc[:,11]

# === 2. Standardize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Apply PCA ===
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# === 4. Plot explained variance ===
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Reduce to 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# === 5. Visualize 2D PCA ===
plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=true_labels, cmap='viridis', s=50)
plt.title('PCA Projection to 2D Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.tight_layout()
plt.show()

# === 6. Export full PCA-transformed data ===
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_columns)
pca_df['True_Label'] = true_labels

pca_df.to_csv('pca_transformed_data.csv', index=False)
