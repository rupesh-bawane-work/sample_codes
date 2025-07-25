import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# === 1. Load dataset ===
df = pd.read_csv(...)
X = df[:,:10]
y = df[:,11]

# === 2. Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 4. One-hot encode labels ===
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# === 5. Build ANN model ===
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# === 6. Compile and train model ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=50, batch_size=8, verbose=0)

# === 7. Save the trained model ===
model.save('ann_model.h5')

# === 8. Make predictions ===
train_probs = model.predict(X_train)
test_probs = model.predict(X_test)
train_preds = np.argmax(train_probs, axis=1)
test_preds = np.argmax(test_probs, axis=1)

# === 9. Apply PCA for 2D visualization ===
pca = PCA(n_components=2)
X_all = np.vstack((X_train, X_test))
X_pca_all = pca.fit_transform(X_all)
X_train_pca = X_pca_all[:len(X_train)]
X_test_pca = X_pca_all[len(X_train):]

# === 10. Visualize predictions in PCA space ===
# Train
plt.figure(figsize=(7, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_preds, cmap='viridis', s=50)
plt.title('ANN Train Predictions (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Test
plt.figure(figsize=(7, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_preds, cmap='viridis', s=50, edgecolors='k')
plt.title('ANN Test Predictions (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# === 11. Evaluate model ===
print("Classification Report (Test Set):")
print(classification_report(y_test, test_preds))
print(f"Accuracy: {accuracy_score(y_test, test_preds):.2f}")

# === 12. Export predictions to CSV ===
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
final_df.to_csv('ann_classification_with_pca.csv', index=False)

