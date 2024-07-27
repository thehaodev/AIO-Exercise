import numpy as np
import math


def cal_mean_variance(data: np.array):
    mean = np.mean(data)
    variance = np.var(data)

    return mean, variance


def gauss_function(m, var, x):
    e_hat = np.exp(-1 * math.pow((x-m), 2) / (2*var))
    f_x = e_hat / (math.sqrt(var) * math.sqrt(2*math.pi))

    return f_x


def simple_gaussian():

    data = np.array([
        [1.4, 0],
        [1.0, 0],
        [1.3, 0],
        [1.9, 0],
        [2.0, 0],
        [1.8, 0],
        [3.0, 1],
        [3.8, 1],
        [4.1, 1],
        [3.9, 1],
        [4.2, 1],
        [3.4, 1],
    ], dtype=np.float64)

    data_class_0 = data[data[:, 1] == 0]
    data_class_1 = data[data[:, 1] == 1]

    m_0, var_0 = cal_mean_variance(data=data_class_0[:, 0])
    m_1, var_1 = cal_mean_variance(data=data_class_1[:, 0])

    print(f"mean(0) = {m_0}, variance(0) = {var_0}")
    print(f"mean(1) = {m_1}, variance(1) = {var_1}")

    x = 3.4
    p_0 = data_class_0.shape[0] / data.shape[0]
    p_1 = data_class_1.shape[0] / data.shape[0]
    p_len_0 = gauss_function(m_0, var_0, x)
    p_len_1 = gauss_function(m_1, var_1, x)

    print(f"P(Class = 0 | X) = {p_len_0*p_0}")
    print(f"P(Class = 1 | X) = {p_len_1*p_1}")


FEATURE = {
    "Sepal length": 0,
    "Sepal width": 1,
    "Petal length": 2,
    "Petal width": 3,
}


def cal_likelihood(data: np.array, p, e_occur):
    col_p_index = data.shape[1] - 1
    p_arr = data[data[:, col_p_index] == p]
    p_arr_numb = np.delete(p_arr, col_p_index, 1).astype(float)

    result = 1
    for e in e_occur:
        f_space = p_arr_numb[:, FEATURE[e]]
        m, var = cal_mean_variance(f_space)
        result *= gauss_function(m, var, e_occur[e])

    return result


def multi_gaussian():
    data = np.loadtxt("iris.data.txt", delimiter=",", dtype=str)
    unique, counts = np.unique(data[:, data.shape[1] - 1], return_counts=True)
    predict_dict = dict(zip(unique, counts))
    event_occur = {
        "Sepal length": 4.9,
        "Sepal width": 3.1,
        "Petal length": 1.5,
        "Petal width": 0.1
    }

    dict_p_len_c = {}
    for p in predict_dict:
        prior = predict_dict[p] / data.shape[0]
        likelihood = cal_likelihood(data, p, event_occur)
        dict_p_len_c[p] = likelihood*prior

    max_p = max(dict_p_len_c, key=dict_p_len_c.get, default=None)
    print(max_p, dict_p_len_c[max_p])


def run():
    simple_gaussian()


multi_gaussian()
