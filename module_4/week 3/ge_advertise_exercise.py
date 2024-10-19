import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)


def load_data_from_file(fileName="../data/advertising.csv"):
    data = np.genfromtxt(fileName, dtype=None, delimiter=',', skip_header=1)
    features_x = data[:, :3]
    sales_y = data[:, 3]
    features_x = np.c_[np.ones((len(features_x), 1)), features_x]

    return features_x, sales_y


def create_individual(n=4, bound=10):
    return [random.SystemRandom().uniform(-bound / 2, bound / 2) for _ in range(n)]


def compute_loss(individual, x, y):
    theta = np.array(individual)
    y_hat = x.dot(theta)
    loss = np.multiply((y_hat - y), (y_hat - y)).mean()
    return loss


def compute_fitness(individual, x, y):
    loss = compute_loss(individual, x, y)
    fitness = 1 / (loss + 1)
    return fitness


def mutate(individual, mutation_rate=0.05):
    return [1 - val if random.SystemRandom().random() < mutation_rate else val for val in individual]


def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()
    rd = random.SystemRandom().random()
    for i in range(len(individual1_new)):
        if rd < crossover_rate:
            individual1_new[i] = individual1[i]
            individual2_new[i] = individual2[i]

    return individual1_new, individual2_new


def selection(sorted_old_population, m=100):
    index1 = random.SystemRandom().randint(0, m - 1)
    while True:
        index2 = random.SystemRandom().randint(0, m - 1)
        if index2 != index1:
            break
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


def create_new_population(old_population, x, y, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=lambda individual: compute_fitness(individual, x, y))

    if gen % 1 == 0:
        print(" Best loss :", compute_loss(sorted_population[m - 1], x, y), " with chromsome : ",
              sorted_population[m - 1])

    new_population = []
    while len(new_population) < m - elitism:
        parent1, parent2 = selection(sorted_population), selection(sorted_population)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    # Elitism: retain the best individuals
    new_population = sorted_population[:elitism]

    return new_population, compute_loss(sorted_population[m - 1], x, y)


def run():
    x, y = load_data_from_file()

    # Question 1: A
    # Question 2:A
    print(x[:5, :])

    # Question 3:B
    print(y.shape)

    # Question 4-> C
    individual = [4.09, 4.82, 3.10, 4.02]
    fitness_score = compute_fitness(individual, x, y)
    print(fitness_score)

    # question 5 -> A
    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    individual1, individual2 = crossover(individual1, individual2, 2.0)
    print(" individual1 : ", individual1)
    print(" individual2 : ", individual2)

    # question 6 -> A
    before_individual = [4.09, 4.82, 3.10, 4.02]
    after_individual = mutate(individual, mutation_rate=2.0)
    print(before_individual == after_individual)

    # Questions 7 -> A
    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    old_population = [individual1, individual2]
    new_population, _ = create_new_population(old_population, x, y, elitism=2, gen=1)


run()
