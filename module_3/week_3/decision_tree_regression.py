import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from matplotlib import pyplot as plt


def run_simple_test():
    # Importing the dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    x = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    regressor_max_depth_three = DecisionTreeRegressor(random_state=0, max_depth=3)
    # Fit the regressor object to the dataset.
    regressor_max_depth_three.fit(x, y)

    regressor_min_samples_leaf_fourth = DecisionTreeRegressor(random_state=0, min_samples_leaf=4)
    # Fit the regressor object to the dataset.
    regressor_min_samples_leaf_fourth.fit(x, y)

    fig, ax = plt.subplots(figsize=(20, 20))
    tree.plot_tree(regressor_max_depth_three, ax=ax, feature_names=["Level"], filled=True)
    plt.show()


run_simple_test()



