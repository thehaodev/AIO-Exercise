def compute_loss_mse(y_hat, y):
    return (y_hat - y) ** 2


def compute_loss_mae(y_hat, y):
    return abs(y_hat-y)


def compute_gradient_w(xi, y, y_hat):
    dl_dwi = 2 * xi * (y_hat - y)
    return dl_dwi


def compute_gradient_b(y, y_hat):
    dl_db = 2 * (y_hat - y)
    return dl_db


def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db
    return b


def update_weight_w(wi, dl_dwi, lr):
    wi = wi - lr * dl_dwi
    return wi
