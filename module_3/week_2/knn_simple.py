import csv
import numpy as np
import math


def get_train_data():
    file = open("Iris.csv", "r")
    dataset = csv.reader(file)
    dataset = np.array(list(dataset))
    dataset = np.delete(dataset, 0, 0)
    dataset = np.delete(dataset, 0, 1)
    file.close()

    training_set = dataset[:149]
    testing_set = dataset[149:]

    return training_set, testing_set


def compute_distance(point1, point2):
    result = 0
    for i in range(4):
        result += (float(point1[i]) - float(point2[i])) ** 2

    return math.sqrt(result)


def compute_k_nearest_neighbor(training_set, item, k):
    distances = []
    for data_point in training_set:
        distances.append(
            {
                "label": data_point[-1],
                "value": compute_distance(item, data_point)
            }
        )
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]

    return labels[:k]


def vote_the_distances(array):
    labels = set(array)
    result = ""
    max_occur = 0
    for label in labels:
        num = array.count(label)
        if num > max_occur:
            max_occur = num
            result = label

    return result


def run_simple_test():
    training_set, testing_set = get_train_data()
    k = 5
    # print(testingSet)
    for item in testing_set:
        knn = compute_k_nearest_neighbor(training_set, item, k)
        result = vote_the_distances(knn)
        print("GT = ", item[-1], ", Prediction: =", result)
