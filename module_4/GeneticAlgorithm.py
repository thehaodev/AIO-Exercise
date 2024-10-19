import random


def generate_random_value():
    return random.SystemRandom().randint(0, 1)


def compute_fitness(individual):
    return sum(individual)


def create_individual(n):
    return [generate_random_value() for _ in range(n)]


def crossover(ind1, ind2, rate=0.9):
    ind1_new = ind1.copy()
    ind2_new = ind2.copy()
    rd = random.SystemRandom().random()
    for i in range(len(ind1)):
        if rd < rate:
            ind1_new[i] = ind1[i]
            ind2_new[i] = ind2[i]
    return ind1_new, ind2_new


def mutate(individual, rate=0.05):
    return [1 - val if random.SystemRandom().random() < rate else val for val in individual]


# Compare two random gen and choose the better one
def selection(population):
    n_population = len(population)
    idx1 = random.SystemRandom().randint(0, n_population - 1)
    idx2 = random.SystemRandom().randint(0, n_population - 1)

    fitness_score_1 = compute_fitness(population[idx1])
    fitness_score_2 = compute_fitness(population[idx2])

    if fitness_score_1 > fitness_score_2:
        return population[idx1]
    else:
        return population[idx2]
