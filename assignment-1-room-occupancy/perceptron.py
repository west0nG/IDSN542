"""Binary perceptron using the update rule covered in Module 1."""

import numpy as np


class Perceptron:
    def __init__(self, eta=0.01, n_iter=30, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(0.0, 0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            # Shuffle paired training rows, without changing the time split.
            indices = rgen.permutation(len(y))
            epoch_errors = 0
            for i in indices:
                xi, target = X[i], y[i]
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                epoch_errors += int(update != 0.0)
            self.errors_.append(epoch_errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
