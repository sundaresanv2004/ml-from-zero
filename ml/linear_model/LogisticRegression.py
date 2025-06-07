import numpy as np

from Test.MultiLinearRegressionTest import y_pred


class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=100):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-z))

            loss = (-1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

            dW = (1 / m) * np.dot(X.T, (y_pred - y))
            dB = np.mean(y_pred - y)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-z))
        return (y_pred >= 0.5).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
