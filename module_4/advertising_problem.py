import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, index):
    # Check index in range
    if index < 0 or index >= len(data[0]):
        raise IndexError(f"Column index {index} is out of range.")

    column_data = [row[index] for row in data]
    return column_data


def predict(x_features, weights):
    return sum([f * w for f, w in zip(x_features, weights)])


def compute_loss(y_hat, y):
    return (y_hat - y) ** 2


def compute_gradient_w(x_features, y, y_hat):
    dl_dweights = [2 * xi * (y_hat - y) for xi in x_features]
    return dl_dweights


def update_weight(weights, dl_dweights, lr):
    weights = [w - lr * dw for w, dw in zip(weights, dl_dweights)]
    return weights


def initialize_params():
    return [0, 0.016992259082509283, 0.0070783670518262355, -0.002307860847821344]


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    # get tv (index=0)
    tv_data = get_column(data, 0)

    # get radio (index=1)
    radio_data = get_column(data, 1)

    # get newspaper (index=2)
    newspaper_data = get_column(data, 2)

    # get sales (index=3)
    sales_data = get_column(data, 3)

    # building X input  and y output for training
    # Create list of features for input
    x = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return x, y


def implement_linear_regression(x_feature, y_ouput, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    n = len(y_ouput)
    for epoch in range(epoch_max):
        print("epoch", epoch)
        for i in range(n):
            # get a sample - row i
            features_i = x_feature[i]
            y = y_ouput[i]

            # compute output
            y_hat = predict(features_i, weights)

            # compute loss
            loss = compute_loss(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_dweights = compute_gradient_w(features_i, y, y_hat)

            # update parameters
            weights = update_weight(weights, dl_dweights, lr)

            # logging
            losses.append(loss)
    return weights, losses


def run():
    # Question 12 -> A
    x, y = prepare_data('advertising.csv')
    _, loss = implement_linear_regression(x, y)
    print(loss[9999])

    plt.plot(loss[-100:])
    plt.xlabel("#iteration")
    plt.ylabel("MSE loss")
    plt.show()


run()
