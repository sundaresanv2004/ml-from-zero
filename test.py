import numpy as np
import matplotlib.pyplot as plt
from linear_model.LinearRegression import SimpleLinearRegression, MultiLinearRegression
from sklearn.preprocessing import StandardScaler

"""
----- SimpleLinearRegression -----
# Simulated data: Years of experience (x), Salary in $1000s (y)
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
y_train = np.array([45, 50, 54, 60, 63, 70, 74, 78, 85, 95], dtype=np.float64)

# Create and train model
model = SimpleLinearRegression()
cost_history = model.fit(x_train, y_train, epochs=1000)

# Predict on training data
pred = model.predict(x_train)

# Print weights and bias
print(f"Weight (slope): {model.coef_}")
print(f"Bias (intercept): {model.intercept_}")

# Plot results
plt.scatter(x_train, y_train, label="Actual")
plt.plot(x_train, pred, color="red", label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($1000s)")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()

# Plot cost history
plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.show()

print(model.predict(1.5))
"""

# ----- MultiLinearRegression -----

# X: features (TV, Radio, Newspaper)
X = np.array([
    [230.1, 37.8, 69.2],
    [44.5, 39.3, 45.1],
    [17.2, 45.9, 69.3],
    [151.5, 41.3, 58.5],
    [180.8, 10.8, 58.4],
    [8.7, 48.9, 75.0],
    [57.5, 32.8, 23.5],
    [120.2, 19.6, 11.6],
    [8.6, 2.1, 1.0],
    [199.8, 2.6, 21.2]
])

# y: target (Sales)
y = np.array([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = MultiLinearRegression()
history = model.fit(X_scaled, y, epochs=1000)

# Predict on training data
y_pred = model.predict(X_scaled)

# Predict on new (unnormalized) sample — normalize it first!
new_data = np.array([[150, 20, 25]])
new_data_scaled = scaler.transform(new_data)  # Apply same scaling
prediction = model.predict(new_data_scaled)

# Output
print("Predicted Sales for [150, 20, 25]:", prediction)

# Plot cost history
plt.plot(history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.grid(True)
plt.show()

# Optional: Actual vs Predicted Scatter Plot
plt.scatter(y, y_pred)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
