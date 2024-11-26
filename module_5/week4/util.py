import numpy as np


def df_w(w):
    w1, w2 = w  # Giải nén W thành w1 và w2
    dw1 = 0.2 * w1  # Tính đạo hàm theo w1
    dw2 = 4 * w2  # Tính đạo hàm theo w2
    d_w = np.array([dw1, dw2])  # Tạo mảng gradient
    return d_w  # Trả về gradient
