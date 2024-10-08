import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, index):
    # Check index in range
    if index < 0 or index >= len(data[0]):
        raise IndexError(f"Column index {index} is out of range.")

    column_data = [row[index] for row in data]
    return column_data


def implement_linear_regression(x_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()

    n = len(y_data)
    for _ in range(epoch_max):
        for i in range(n):
            # get a sample
            x1 = x_data[0][i]
            x2 = x_data[1][i]
            x3 = x_data[2][i]

            y = y_data[i]

            # print(y)
            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # compute loss
            loss = compute_loss_mse(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            # logging
            losses.append(loss)
    return w1, w2, w3, b, losses


def implement_linear_regression_nsamples(x_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    n = len(y_data)

    for _ in range(epoch_max):

        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0

        for i in range(n):
            # get a sample
            x1 = x_data[0][i]
            x2 = x_data[1][i]
            x3 = x_data[2][i]

            y = y_data[i]

            # print(y)
            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # compute loss
            loss = compute_loss_mae(y, y_hat)
            loss_total = loss_total + loss

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # accumulate
            dw1_total = dw1_total + dl_dw1
            dw2_total = dw2_total + dl_dw2
            dw3_total = dw3_total + dl_dw3
            db_total = db_total + dl_db

        # (after processing N samples) - update parameters
        w1 = update_weight_wi(w1, dl_dw1/n, lr)
        w2 = update_weight_wi(w2, dl_dw2/n, lr)
        w3 = update_weight_wi(w3, dl_dw3/n, lr)
        b = update_weight_b(b, dl_db/n, lr)

        # logging
        losses.append(loss_total / n)

    return w1, w2, w3, b, losses


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    # get tv index = 0
    tv_data = get_column(data, 0)

    # get radio index = 1
    radio_data = get_column(data, 1)

    # get newspaper index = 2
    newspaper_data = get_column(data, 2)

    # get sales index = 3
    sales_data = get_column(data, 3)

    # Building X input and y ouput for training
    x = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return x, y


def predict(x1, x2, x3, w1, w2, w3, b):
    result = w1 * x1 + w2 * x2 + w3 * x3 + b
    return result


def compute_loss_mse(y_hat, y):
    return (y_hat - y) ** 2


def compute_loss_mae(y_hat, y):
    return abs(y_hat-y)


def compute_gradient_wi(xi, y, y_hat):
    dl_dwi = 2 * xi * (y_hat - y)
    return dl_dwi


def compute_gradient_b(y, y_hat):
    dl_db = 2 * (y_hat - y)
    return dl_db


def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr * dl_dwi
    return wi


def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db
    return b


def initialize_params():
    # In real practice we use w = random.gauss(mu=0.0, sigma=0.01)
    # b  = 0
    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b


def run_simple():
    # Question 1 -> A
    x, y = prepare_data('advertising.csv')
    list_data = [sum(x[0][:5]), sum(x[1][:5]), sum(x[2][:5]), sum(y[:5])]
    print(list_data)

    # Question 2 -> A
    y_p = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(y_p)

    # Question 3 -> A
    loss = compute_loss_mse(y_hat=1, y=0.5)
    print(loss)

    # Question 4 -> A
    g_wi = compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
    print(g_wi)

    # Question 5 -> B
    g_b = compute_gradient_b(y=2.0, y_hat=0.5)
    print(g_b)

    # Question 6 -> A
    after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
    print(after_wi)

    # Question 7 -> A
    after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
    print(after_b)

    (w1, w2, w3, b, losses) = implement_linear_regression(x, y)
    plt.plot(losses[:100])
    plt.xlabel("#iteration")
    plt.ylabel("Loss")
    plt.show()

    # Question 8->A
    print(w1, w2, w3)

    # Question 9 -> B
    tv = 19.2
    radio = 35.9
    newspaper = 51.3
    sales = predict(tv, radio, newspaper, w1, w2, w3, b)
    print(f'predicted sales is {sales}')

    # Question 10 -> A
    loss = compute_loss_mae(y_hat=1, y=0.5)
    print(loss)

    # Question 11 -> D
    w1, w2, w3, _, _ = implement_linear_regression_nsamples(x, y, 1000)
    print(w1, w2, w3)


run_simple()
