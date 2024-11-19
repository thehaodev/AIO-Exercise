import numpy as np
import matplotlib.pyplot as plt
from RegressionInterface import RegressionInterface


class LogisticRegression(RegressionInterface):
    def __init__(self, x_data, y_target, theta, batch_size, learning_rate=0.01, num_epochs=10000):
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

    def predict(self, x_data):
        z = np.dot(x_data, self.theta)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, x_data, y_data):
        y_pred = self.predict(x_data)
        return (-y_data.T.dot(np.log(y_pred)) - (1 - y_data).T.dot(np.log(1 - y_pred))) / y_data.size

    def compute_gradient(self, x_data, y_data):
        y_pred = self.predict(x_data)
        return np.dot(x_data.T, (y_pred - y_data)) / y_data.size

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

        # Compute accuracy
        self.accuracy.append(self.compute_accuracy(x_data, y_data))
        print(loss)

    def stochastic_training(self):
        for _ in range(self.num_epochs):
            for i in range(0, self.num_samples, self.batch_size):
                x_i = self.x_data[i]
                y_i = self.y_target[i]

                self.training(x_i, y_i)

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

    def compute_accuracy(self, x_data, y_data):
        y_hat = self.predict(x_data).round()
        self.accuracy.append((y_hat == y_data).mean())

    def accuracy_model(self):
        return len(self.accuracy) / self.y_target.size

    def plot_loss_accuracy(self):
        _, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(self.losses)
        ax[0].set(xlabel='Epoch', ylabel='Loss')
        ax[0].set_title('Loss')

        ax[1].plot(self.accuracy)
        ax[1].set(xlabel='Epoch', ylabel='Accuracy')
        ax[1].set_title('Accuracy')

        plt.show()
