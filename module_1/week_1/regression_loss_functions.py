import math
import random


def calc_loss_functions():
    n = input("Input n = ")
    loss_name = input("Input loss name = ")
    functions = ["MAE", "MSE", "RMSE"]
    if not n.isnumeric():
        return print("number of samples must be an integer number")

    n_int = int(n)
    if loss_name == functions[0]:
        mae_loss = 0
        for i in range(1, n_int):
            target = random.uniform(0, 10)
            predict = random.uniform(0, 10)
            loss = (1/n_int) * abs(target-predict)
            mae_loss += loss
            print(f"loss name: {loss_name}, sample: {i}, pred: {predict}, target: {target}, loss: {loss}")

        return print(f"final {mae_loss}")

    if loss_name == functions[1]:
        mse_loss = 0
        for i in range(1, n_int):
            target = random.uniform(0, 10)
            predict = random.uniform(0, 10)
            loss = (1/n_int) * pow(target-predict, 2)
            mse_loss += loss
            print(f"loss name: {loss_name}, sample: {i}, pred: {predict}, target: {target}, loss: {loss}")

        return print(f"final {mse_loss}")

    if loss_name == functions[2]:
        mse_loss = 0
        for i in range(1, n_int):
            target = random.uniform(0, 10)
            predict = random.uniform(0, 10)
            loss = (1/n_int) * pow(target-predict, 2)
            mse_loss += loss
            print(f"loss name: {loss_name}, sample: {i}, pred: {predict}, target: {target}, loss: {loss}")
        rmse_loss = math.sqrt(mse_loss)

        return print(f"final {rmse_loss}")


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True
