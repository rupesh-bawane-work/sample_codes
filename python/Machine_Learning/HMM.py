import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === 1. Create or load sequential data ===
df = pd.read_csv(...)
X = df.iloc[:,:10]
y = df.iloc[:,11]

# === 2. Standardize data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Fit Gaussian HMM ===
n_states = 3
hmm_model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100, random_state=42)
hmm_model.fit(X_scaled)

# === 4. Predict hidden states ===
hidden_states = hmm_model.predict(X_scaled)
df['Hidden_State'] = hidden_states

# === 5. PCA for 2D visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# === 6. Visualize PCA results colored by hidden state ===
plt.figure(figsize=(8, 5))
for state in range(n_states):
    mask = hidden_states == state
    plt.plot(X_pca[mask, 0], X_pca[mask, 1], '.', label=f'State {state}')
plt.title('HMM Hidden States (PCA Projection)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 7. Export to CSV ===
df.to_csv('hmm_hidden_states_with_pca.csv', index=False)

