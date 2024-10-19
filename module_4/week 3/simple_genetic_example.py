from module_4 import GeneticAlgorithm as gA


def run():
    n = 100  # Length of each individual (binary string)
    m = 100  # Population size
    generations = 100  # Number of generations
    elitism = 2  # Number of elite individuals to carry over

    # Step 1: Create individual with random
    population = [gA.create_individual(n) for _ in range(m)]
    fitness_history = []

    for gen in range(generations):
        # Step 2: Selection
        # Track the best fitness in this generation
        population = sorted(population,
                            key=gA.compute_fitness,
                            reverse=True)
        if gen % 10 == 0:
            best_fitness = gA.compute_fitness(population[0])
            fitness_history.append(best_fitness)
            print(f"Generation {gen} - Best Number of 1s: {best_fitness}")

        # Elitism: retain the best individuals
        new_population = population[:elitism]

        # Mutate two gen
        while len(new_population) < m:
            parent1, parent2 = gA.selection(population), gA.selection(population)
            child1, child2 = gA.crossover(parent1, parent2)
            new_population.append(gA.mutate(child1))
            new_population.append(gA.mutate(child2))

        population = new_population[:m]

    best_individual = sorted(population,
                             key=gA.compute_fitness,
                             reverse=True)[0]
    print("Best individual:", best_individual)
    print("Best fitness (number of 1s):", gA.compute_fitness(best_individual))
    print("Fitness history:", fitness_history)


run()
