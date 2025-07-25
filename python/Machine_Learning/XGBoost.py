import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df[:,:10]
y = df[:,11]

# === 2. Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. Train XGBoost model ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)

# === 5. Make predictions ===
train_preds = xgb.predict(X_train)
test_preds = xgb.predict(X_test)

# === 6. Apply PCA for 2D visualization ===
pca = PCA(n_components=2)
X_all = np.vstack((X_train, X_test))
X_pca_all = pca.fit_transform(X_all)
X_train_pca = X_pca_all[:len(X_train)]
X_test_pca = X_pca_all[len(X_train):]

# === 7. Visualize PCA + Predictions ===
# Train
plt.figure(figsize=(7, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_preds, cmap='viridis', s=50)
plt.title('XGBoost Train Predictions (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Test
plt.figure(figsize=(7, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_preds, cmap='viridis', s=50, edgecolors='k')
plt.title('XGBoost Test Predictions (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# === 8. Evaluate model ===
print("Classification Report (Test Set):")
print(classification_report(y_test, test_preds))
print(f"Accuracy: {accuracy_score(y_test, test_preds):.2f}")

# === 9. Export results to CSV ===
train_df = X_train
train_df['True_Label'] = y_train.values
train_df['Predicted_Label'] = train_preds
train_df['Set'] = 'Train'
train_df['PCA1'] = X_train_pca[:, 0]
train_df['PCA2'] = X_train_pca[:, 1]

test_df = X_test
test_df['True_Label'] = y_test.values
test_df['Predicted_Label'] = test_preds
test_df['Set'] = 'Test'
test_df['PCA1'] = X_test_pca[:, 0]
test_df['PCA2'] = X_test_pca[:, 1]

final_df = pd.concat([train_df, test_df])
final_df.to_csv('xgboost_classification_with_pca.csv', index=False)

