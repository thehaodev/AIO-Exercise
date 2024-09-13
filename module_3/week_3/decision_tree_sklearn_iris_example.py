from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn import tree


def run_sklearn_iris_example():
    dataset = load_iris()
    x = dataset.data
    y = dataset.target
    print(y.shape)
    classifier = tree.DecisionTreeClassifier(criterion="gini",
                                             max_depth=4, min_samples_leaf=10)
    classifier.fit(x, y)
    _, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classifier, ax=ax, feature_names=["sepal length", "sepal width",
                                                     "petal length", "petal width"],
                   filled=True)
    plt.show()


run_sklearn_iris_example()
