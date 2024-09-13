def frequency(a, x):
    count = 0
    for i in a:
        if i == x:
            count += 1
    return count


def gini_impurity(array_data):
    p_sum = 0
    n = len(array_data)
    for i in set(array_data):
        p_sum = p_sum + (frequency(array_data, i) / n) * (frequency(array_data, i) / n)
    gini = 1 - p_sum
    return gini
