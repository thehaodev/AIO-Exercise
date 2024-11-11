import numpy as np


class SoftmaxRegression:
    def __init__(self, x_data, y_target, theta, batch_size, learning_rate, num_epochs):
        self.num_samples = x_data.shape[0]
        self.x_data = x_data
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initial weights
        self.theta = theta
        self.losses = []
        self.accuracy = []

    def softmax_function(self, x_data):
        z = np.dot(x_data, self.theta)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1)[:, None]

    def predict(self, x_data):
        return self.softmax_function(x_data)

    def compute_loss(self, x_data, y_data):
        y_hat = self.softmax_function(x_data)
        return (-1 / self.num_samples) * np.sum(y_data * np.log(y_hat))

    def compute_gradient(self, x_data, y_data):
        y_hat = self.softmax_function(x_data)
        return np.dot(x_data.T, (y_hat - y_data)) / self.num_samples

    def update_gradient(self, x_data, y_data):
        gradient = self.compute_gradient(x_data, y_data)
        self.theta = self.theta - self.learning_rate * gradient

    def training(self, x_data, y_data):
        # compute loss
        loss = self.compute_loss(x_data, y_data)

        # compute and update gradient
        self.update_gradient(x_data, y_data)

        # Compute losses
        self.losses.append(loss)

        print(loss)

    def mini_batch_training(self):
        mini_batch_loss = []
        for _ in range(self.num_epochs):
            for i in range(0, self.x_data.shape[0], self.batch_size):
                x_i = self.x_data[i:i + self.batch_size]
                y_i = self.y_target[i:i + self.batch_size]
                self.training(x_i, y_i)
            train_batch_losses = sum(self.losses) / len(self.losses)
            mini_batch_loss.append(train_batch_losses)

        self.losses = mini_batch_loss

    def batch_training(self):
        batch_loss = []
        for _ in range(self.num_epochs):
            self.training(self.x_data, self.y_target)
            batch_losses = sum(self.losses) / len(self.losses)
            batch_loss.append(batch_losses)

        self.losses = batch_loss
