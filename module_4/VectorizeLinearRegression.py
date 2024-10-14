import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict(x, theta):
    return x.dot(theta)


def compute_loss(y_hat, y):
    n = y.shape[0]
    return np.sum((y_hat - y) ** 2) / n


def compute_gradient(y_hat, y, x):
    n = y.shape[0]
    k = 2 * (y_hat - y)
    return x.T.dot(k) / n


def update_gradient(theta, gradient, lr):
    theta = theta - lr * gradient
    return theta


def prepare_data(data_frame):
    x, y = data_frame.iloc[:, :-1].to_numpy(), data_frame.iloc[:, -1:].to_numpy()

    # get vectorize for x feature
    x_bias, _, _, _ = mean_normalization(x)

    # Create theta
    b = 0
    rng = np.random.default_rng(seed=42)
    number_feature = x.shape[1]
    mu = 0.0  # mean
    sigma = 0.0001  # standard deviation
    weight = rng.normal(mu, sigma, number_feature)
    theta = np.array(np.append(b, weight))

    return x_bias, y, theta[:, np.newaxis]


def visualize(data):
    plt.plot(data)
    plt.title('Loss over epoch')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.show()


def training(x_bias, y, theta):
    epoch_max = 100
    lr = 0.01  # Learning rate
    losses = []

    # Train
    for _ in range(epoch_max):
        # Predict
        y_hat = predict(x_bias, theta)

        # Compute MSE (loss)
        loss = compute_loss(y_hat, y)
        losses.append(loss)

        # Compute gradient
        gradient = compute_gradient(y_hat, y, x_bias)

        # Update theta
        theta = update_gradient(theta, gradient, lr)

    final_theta = theta

    return losses, final_theta


def mean_normalization(x):
    n = len(x)
    maxi = np.max(x)
    mini = np.min(x)
    avg = np.mean(x)
    x = (x - avg) / (maxi - mini)
    x_b = np.c_[np.ones((n, 1)), x]
    return x_b, maxi, mini, avg


def run():
    data_advertising = pd.read_csv("data/advertising.csv")

    x_bias, y, theta = prepare_data(data_advertising)
    losses, _ = training(x_bias, y, theta)
    visualize(losses)
