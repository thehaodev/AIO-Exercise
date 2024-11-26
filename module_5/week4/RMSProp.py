import numpy as np
import util


def rms_prop(w, d_w, lr, s, gamma, epsilon):
    s = gamma * s + (1 - gamma) * d_w ** 2
    adapt_lr = lr / (np.sqrt(s) + epsilon)
    w = w - adapt_lr * d_w
    return w, s


def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2])
    S = np.array([0, 0])
    gamma = 0.9
    epsilon = 1e-6
    results = [W.copy()]  # Sử dụng copy để lưu kết quả

    for _ in range(epochs):  # Vòng lặp for
        d_w = util.df_w(W)  # Tính gradient
        W, S = optimizer(W, d_w, lr, S, gamma=gamma, epsilon=epsilon)  # Cập nhật trọng số
        results.append(W.copy())  # Lưu kết quả

    return results


def run():
    lr = 0.3  # Hệ số học
    epochs = 30  # Số epochs

    # In ra kết quả sau 30 epochs
    train_result = train_p1(rms_prop, lr, epochs)
    print(train_result[-1])


run()
