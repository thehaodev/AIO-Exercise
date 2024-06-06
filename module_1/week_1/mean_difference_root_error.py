def md_nre_sample_loss(y, y_hat, n, p):
    difference = pow(y, 1/n) - pow(y_hat, 1/n)
    return print(pow(difference, p))
