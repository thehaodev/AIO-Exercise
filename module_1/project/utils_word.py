import numpy


def load_vocab(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = sorted(set([line.strip().lower() for line in lines]))

        return words


def levenshtein_distance(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1-1] == token2[t2-1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                distances[t1][t2] = compare(a, b, c)

    return distances[len(token1)][len(token2)]


def compare(a, b, c):
    if a <= b and a <= c:
        return a + 1
    elif b <= a and b <= c:
        return b + 1
    else:
        return c + 1
