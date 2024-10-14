import numpy as np
import matplotlib.pyplot as plt
import random
from module_4 import VectorizeLinearRegression as vLR


def stochastic_gradient_descent(x_b, y, n, n_epochs, learning_rate):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033],
                         [0.29763545]])

    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):
        for i in range(n):
            random_index = i
            xi = x_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            # Predict
            y_hat = xi.dot(thetas)

            # Compute loss
            loss = (y_hat - yi) * (y_hat - yi) / 2

            # Compute gradient
            gradient_loss = (y_hat - yi)
            gradients = xi.T.dot(gradient_loss)

            # Update theta
            thetas = thetas - learning_rate * gradients

            thetas_path.append(thetas)
            losses.append(loss[0][0])

    return thetas_path, losses


def mini_batch_gradient_descent(x_b, y, n, n_epochs, learning_rate, minibatch_size):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033],
                         [0.29763545]])

    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):
        shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3,
                                       132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16,
                                       185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126,
                                       165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190,
                                       169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131,
                                       77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139,
                                       195, 72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147,
                                       92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47,
                                       174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67,
                                       129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24,
                                       168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55,
                                       133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122,
                                       154])

        x_b_shuffled = x_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, n, minibatch_size):
            xi = x_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]

            # Predict
            y_hat = xi.dot(thetas)

            # Compute loss
            loss = (y_hat - yi) * (y_hat - yi) / 2

            # Compute gradient
            gradient_loss = (y_hat - yi) / minibatch_size
            gradients = xi.T.dot(gradient_loss)

            # Update theta
            thetas = thetas - learning_rate * gradients
            thetas_path.append(thetas)

            loss_mean = np.sum(loss) / minibatch_size
            losses.append(loss_mean)

    return thetas_path, losses


def batch_gradient_descent(x_b, y, n_epochs, learning_rate):
    thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033],
                         [0.29763545]])

    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):
        y_hat = vLR.predict(x_b, thetas)

        # Compute MSE (loss)
        loss = vLR.compute_loss(y_hat, y)
        losses.append(loss)

        # Compute gradient
        gradient = vLR.compute_gradient(y_hat, y, x_b)

        # Update theta
        thetas = vLR.update_gradient(thetas, gradient, learning_rate)

    return thetas_path, losses


def run():
    data = np.genfromtxt('../data/advertising.csv', delimiter=',', skip_header=1)
    n = data.shape[0]
    x = data[:, :3]
    y = data[:, 3:]

    x_b, _, _, _ = vLR.mean_normalization(x)

    # Question 1 -> D
    _, sgd_losses = stochastic_gradient_descent(x_b, y, n,
                                                n_epochs=1, learning_rate=0.01)
    print(np.sum(sgd_losses))

    # Visualize sgd_losses
    plt.figure(1)
    plt.plot(sgd_losses, color="r")

    # Question 2 -> D
    _, mbgd_losses = mini_batch_gradient_descent(x_b, y, n,
                                                 n_epochs=50, minibatch_size=20, learning_rate=0.01)
    print(np.sum(mbgd_losses))

    # Visualize mbgd_losses
    plt.figure(2)
    plt.plot(mbgd_losses, color="r")

    # Question 3 ->
    _, bgd_losses = batch_gradient_descent(x_b, y, n_epochs=100, learning_rate=0.01)
    print(np.sum(bgd_losses))

    plt.figure(3)
    plt.plot(bgd_losses, color="r")

    plt.show()


run()
