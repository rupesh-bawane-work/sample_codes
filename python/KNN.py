# KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load the cleaned dataset
df = pd.read_csv(...........)
X = df[:,:10]
y = df[:,11]
target_names = df.target_names

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Split the reduced data
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(8, 6))
for i, label in enumerate(target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], label=label, alpha=0.6)

# Highlight test samples
plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolors='black', linewidths=1.5, s=100, label='Test Samples')

plt.title("KNN Classification (2D PCA View)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
