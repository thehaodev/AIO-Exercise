import numpy as np
import util


def df_w(w):
    w1, w2 = w
    dw1 = 0.2 * w1
    dw2 = 4 * w2
    return np.array([dw1, dw2])


def adam_optimizer(w, d_w, v, s, lr, beta1, beta2, epsilon, epoch):
    v = beta1 * v + (1 - beta1) * d_w
    s = beta2 * s + (1 - beta2) * d_w**2

    v_corr = v / (1 - beta1**(epoch + 1))
    s_corr = s / (1 - beta2**(epoch + 1))

    w = w - lr * v_corr / (np.sqrt(s_corr) + epsilon)  # Cập nhật W
    return w, v, s


def train_adam(lr, epochs):
    W = np.array([-5.0, -2.0])  # Khởi tạo trọng số
    V = np.zeros(2)  # Khởi tạo V
    S = np.zeros(2)  # Khởi tạo S
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-6
    results = []

    for epoch in range(epochs):
        d_w = util.df_w(W)  # Tính gradient
        W, V, S = adam_optimizer(W, d_w, V, S, lr, beta1, beta2, epsilon, epoch)  # Cập nhật W
        results.append(W.copy())  # Lưu kết quả

    return results


def run():
    lr = 0.2  # Hệ số học
    epochs = 30  # Số epochs

    # In ra kết quả sau 30 epochs
    train_result = train_adam(lr, epochs)
    print(train_result[-1])


run()
