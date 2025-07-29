import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# === 1. Load dataset ===
X = pd.read_csv(...)
X = df[:,:10]
y = df[:,11]

# === 2. Split into train and test ===
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# === 3. Elbow Method on Train ===
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method (Train Set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# === 4. Apply KMeans with selected k ===
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_train)

# Predict cluster labels
train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)

# === 5. Apply PCA to all data for 2D visualization ===
pca = PCA(n_components=2)
X_all = pd.concat([X_train, X_test])
X_pca = pca.fit_transform(X_all)

# Split PCA results back to train/test
X_train_pca = X_pca[:len(X_train)]
X_test_pca = X_pca[len(X_train):]

# === 6. Visualization ===

# Train set plot
plt.figure(figsize=(7, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_labels, cmap='viridis', s=50)
plt.title('KMeans Clustering (Train Set - PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Test set plot
plt.figure(figsize=(7, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_labels, cmap='viridis', s=50, edgecolors='k')
plt.title('KMeans Clustering (Test Set - PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# === 7. Export Results ===

# Create full labeled DataFrame
train_df = X_train.copy()
train_df['Cluster'] = train_labels
train_df['Set'] = 'Train'

test_df = X_test.copy()
test_df['Cluster'] = test_labels
test_df['Set'] = 'Test'

# Add PCA components
train_df['PCA1'] = X_train_pca[:, 0]
train_df['PCA2'] = X_train_pca[:, 1]

test_df['PCA1'] = X_test_pca[:, 0]
test_df['PCA2'] = X_test_pca[:, 1]

# Combine and export
final_df = pd.concat([train_df, test_df])
final_df.to_csv('kmeans_clustered_with_pca.csv', index=False)
