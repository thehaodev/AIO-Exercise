import gdown
import numpy as np
from gdown.exceptions import FileURLRetrievalError
import pandas as pd


def get_max_sale(table: np.array):
    sales_column = table[:, -1]
    max_sales = sales_column.max()
    index_of_max = np.nonzero(table == max_sales)[0]

    return print(max_sales, index_of_max)


def get_average_tv(table: np.array):
    tv_column = table[:, 0]
    return print(np.mean(tv_column))


def num_of_sale(table: np.array, limit: int):
    sales_column = table[:, -1]
    result = sales_column[np.nonzero(sales_column >= limit)]

    return print(result.size)


def get_mean_radio(table: np.array, limit: int):
    sales_column = table[:, -1]
    radio_column = table[:, 1]
    result = radio_column[np.nonzero(sales_column >= limit)]

    return print(np.mean(result))


def get_scores(table: np.array, first: int, last: int):
    sales_column = table[:, -1]
    sales_mean = np.mean(sales_column)
    score = np.full(sales_column.size, "None")
    score[sales_column > sales_mean] = "Good"
    score[sales_column < sales_mean] = "Bad"
    score[sales_column == sales_mean] = "Average"

    print(score[first:last])


def get_nearest_score(table: np.array, first: int, last: int):
    sales_column = table[:, -1]
    sales_mean = np.mean(sales_column)

    # Find the nearest sales mean
    dif_array = np.absolute(sales_column - sales_mean)
    index_dif = dif_array.argmin()
    near_score = sales_column[index_dif]

    score = np.full(sales_column.size, "None")
    score[sales_column > near_score] = "Good"
    score[sales_column < near_score] = "Bad"
    score[sales_column == near_score] = "Average"

    return print(score[first:last])


def get_sale(table: np.array):
    sales_column = table[:, -1]
    newspaper_column = table[:, -2]
    newspaper_mean = np.mean(newspaper_column)
    result = sales_column[np.nonzero(newspaper_column > newspaper_mean)]

    return print(np.sum(result))


def run():
    url = "https://drive.google.com/uc?id=1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq"
    try:
        gdown.download(url)
    except FileURLRetrievalError:
        print("File cannot download")

    df = pd.read_csv('advertising.csv')
    data = df.to_numpy()

    get_max_sale(data)
    get_average_tv(data)
    num_of_sale(data, 20)
    get_mean_radio(data, 15)
    get_sale(data)
    get_scores(data, 7, 10)
    get_nearest_score(data, 7, 10)


run()
