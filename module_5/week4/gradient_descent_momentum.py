import numpy as np
import util


def sgd_momentum(w, d_w, lr, v, beta):
    V = beta * v + (1 - beta) * d_w
    W = w - lr * V
    return W, V


def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2])
    V = np.array([0, 0])
    beta = 0.5
    results = [W.copy()]  # Sử dụng copy để lưu kết quả

    for _ in range(epochs):  # Thay đổi vòng lặp để phù hợp với Python
        d_w = util.df_w(W)  # Tính gradient
        W, V = optimizer(W, d_w, lr, V, beta=beta)  # Cập nhật trọng số và vận tốc
        results.append(W.copy())  # Lưu kết quả mới

    return results


def run():
    lr = 0.6  # Hệ số học
    epochs = 30  # Số epochs

    # In ra kết quả sau 30 epochs
    train_result = train_p1(sgd_momentum, lr, epochs)
    print(train_result[-1])


run()
