import numpy as np
import matplotlib.pyplot as plt
from ml.linear_model.LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Dataset
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])
y = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(learning_rate=0.1)
model.fit(X_scaled, y, epochs=1000)

# Predict and check accuracy
predictions = model.predict(X_scaled)
accuracy = model.accuracy(X_scaled, y)

print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy:.2f}")
