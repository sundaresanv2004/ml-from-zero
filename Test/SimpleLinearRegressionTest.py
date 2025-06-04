import numpy as np
import matplotlib.pyplot as plt
from ml.linear_model.LinearRegression import SimpleLinearRegression

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

