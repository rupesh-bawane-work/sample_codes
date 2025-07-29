# KNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df.iloc[:,:10]
y = df.iloc[:,:11]

# === 2. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Tune k for KNN ===
k_range = range(1, 11)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o')
plt.title('KNN Accuracy vs. Number of Neighbors (k)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# === 4. Choose best k (based on highest accuracy) ===
optimal_k = k_range[np.argmax(accuracies)]
print(f"Optimal k: {optimal_k}")

# === 5. Train KNN with optimal k and make predictions ===
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

# === 6. Apply PCA for 2D visualization ===
pca = PCA(n_components=2)
X_all = pd.concat([X_train, X_test])
X_pca = pca.fit_transform(X_all)

X_train_pca = X_pca[:len(X_train)]
X_test_pca = X_pca[len(X_train):]

# === 7. Visualization ===

# Train set
plt.figure(figsize=(7, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_preds, cmap='viridis', s=50)
plt.title(f'KNN Train Set (k={optimal_k}) - PCA 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Test set
plt.figure(figsize=(7, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_preds, cmap='viridis', s=50, edgecolors='k')
plt.title(f'KNN Test Set (k={optimal_k}) - PCA 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# === 8. Export results to CSV ===

# Train DataFrame
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['True_Label'] = y_train.values
train_df['Predicted_Label'] = train_preds
train_df['Set'] = 'Train'
train_df['PCA1'] = X_train_pca[:, 0]
train_df['PCA2'] = X_train_pca[:, 1]

# Test DataFrame
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['True_Label'] = y_test.values
test_df['Predicted_Label'] = test_preds
test_df['Set'] = 'Test'
test_df['PCA1'] = X_test_pca[:, 0]
test_df['PCA2'] = X_test_pca[:, 1]

# Combine and export
final_knn_df = pd.concat([train_df, test_df])
final_knn_df.to_csv('knn_classified_with_pca.csv', index=False)

