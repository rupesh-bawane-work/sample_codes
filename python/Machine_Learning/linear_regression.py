import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

# === 1. Load regression dataset ===
df = pd.read_csv(...)
X = df[:,:10]
y = df[:,11]

# === 2. Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. Train Linear Regression model ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === 5. Make predictions ===
train_preds = lr.predict(X_train)
test_preds = lr.predict(X_test)

# === 6. Apply PCA for 2D visualization ===
pca = PCA(n_components=2)
X_all = np.vstack((X_train, X_test))
X_pca_all = pca.fit_transform(X_all)
X_train_pca = X_pca_all[:len(X_train)]
X_test_pca = X_pca_all[len(X_train):]

# === 7. Plot predicted vs actual (test set) ===
plt.figure(figsize=(7, 5))
plt.scatter(y_test, test_preds, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs. Predicted (Test Set)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. Evaluation ===
print("Linear Regression Performance (Test Set):")
print(f"Mean Squared Error: {mean_squared_error(y_test, test_preds):.2f}")
print(f"R² Score: {r2_score(y_test, test_preds):.2f}")

# === 9. Export results to CSV ===
# Combine for export
train_df = pd.DataFrame(X_train, columns=data.feature_names)
train_df['Target'] = y_train.values
train_df['Predicted'] = train_preds
train_df['Set'] = 'Train'
train_df['PCA1'] = X_train_pca[:, 0]
train_df['PCA2'] = X_train_pca[:, 1]

test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df['Target'] = y_test.values
test_df['Predicted'] = test_preds
test_df['Set'] = 'Test'
test_df['PCA1'] = X_test_pca[:, 0]
test_df['PCA2'] = X_test_pca[:, 1]

final_df = pd.concat([train_df, test_df])
final_df.to_csv('linear_regression_with_pca.csv', index=False)

