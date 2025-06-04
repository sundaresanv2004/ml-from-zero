import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.weight = 0
        self.bias = 0
        self.learning_rate = learning_rate

    def fit(self, x, y, epochs=100):
        ...
