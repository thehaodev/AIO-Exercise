import numpy as np
import util


def sgd(w, d_w, lr):
    w = w - lr * d_w  # Cập nhật W theo gradient và tốc độ học
    return w  # Trả về W đã cập nhật


def train_p1(optimizer, lr, epochs):
    W = np.array([-5, -2])
    results = [W.copy()]  # Sử dụng copy để tránh tham chiếu đến mảng gốc

    for _ in range(epochs):  # Thay đổi vòng lặp để phù hợp với Python
        d_w = util.df_w(W)  # Tính gradient
        W = optimizer(W, d_w, lr)  # Cập nhật trọng số
        results.append(W.copy())  # Lưu kết quả

    return results


def run():
    lr = 0.4  # Hệ số học
    epochs = 30  # Số epochs

    # In ra kết quả sau 30 epochs
    train_result = train_p1(sgd, lr, epochs)
    print(train_result[-1])


run()
