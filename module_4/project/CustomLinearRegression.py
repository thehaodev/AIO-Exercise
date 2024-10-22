import numpy as np


class CustomLinearRegression:
    def __init__(self, x_data, y_target, learning_rate=0.01, num_epochs=10000):
        self.num_samples = x_data.shape[0]
        self.x_data = np.c_[np.ones((self.num_samples, 1)), x_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initial weights

        self.theta = np.random.Generator.integers(self.x_data.shape[1], 1)
        self.losses = []

    def compute_loss(self, y_target):
        y_pred = self.predict()
        return np.sum((y_pred - y_target) ** 2) / (y_target.shape[0])

    def r2score(self):
        y_pred = self.predict()
        rss = np.sum((y_pred - self.y_target) ** 2)
        tss = np.sum((self.y_target - self.y_target.mean()) ** 2)
        r2 = 1 - (rss / tss)
        return r2

    def predict(self):
        y_pred = self.x_data.dot(self.theta)
        return y_pred

    def compute_gradient(self):
        y_pred = self.predict()
        n = self.y_target.shape[0]
        k = 2 * (y_pred - self.y_target)
        return self.x_data.T.dot(k) / n

    def update_gradient(self):
        gradient = self.compute_gradient()
        self.theta = self.theta - self.learning_rate * gradient

    def fit(self):
        loss = 0
        for epoch in range(self.num_epochs):
            # Compute loss
            loss += self.compute_loss(self.y_target)
            self.losses.append(loss)

            # Update theta
            self.update_gradient()
            if (epoch % 50) == 0:
                print(f' Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': sum(self.losses) / len(self.losses),
            'weight': self.theta
        }
