import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = 0
        self.bias = 0
        self.learning_rate = learning_rate
        self.predicted_y = None
        self.cost_history = []

    def fit(self, x, y, epochs=100):
        m = len(y)
        prev_cost = float('inf')
        for epoch in range(epochs):
            self.predicted_y = np.dot(x, self.weights) + self.bias

            error = self.predicted_y - y
            cost = (1 / (2 * m)) * np.sum(error**2)
            if (prev_cost - cost) < 1e-6:
                print(f"Early stopping at epoch {epoch}")
                break

            prev_cost = cost
            self.cost_history.append(cost)

            dW = np.mean(error * x)
            dB = np.mean(error)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
            
        return self.cost_history

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    @property
    def coef_(self):
        return self.weights

    @property
    def intercept_(self):
        return self.bias


class MultiLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.cost_history = []

    def fit(self, X, y, epochs=1000):
        m, n = X.shape
        self.weights = np.zeros(n)

        for epoch in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            dW = (1 / m) * np.dot(X.T, error)
            dB = (1 / m) * np.sum(error)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")

        return self.cost_history

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    @property
    def coef_(self):
        return self.weights

    @property
    def intercept_(self):
        return self.bias
