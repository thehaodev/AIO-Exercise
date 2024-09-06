import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def run_test():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    dataset = pd.read_csv(url, names=names)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print(round(accuracy_score(y_test, y_pred), 2))
    print(classification_report(y_test, y_pred))

    error = []

    # Calculating accuracy for K-values between 1 and 30
    for i in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(accuracy_score(pred_i, y_test))

    # Visualize Accuracy vs K Value

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, 30), error, color='blue', marker='o',
             markerfacecolor='yellow', markersize=10)
    plt.title('Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')


run_test()
