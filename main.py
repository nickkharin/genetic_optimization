from genetic import *

if __name__ == '__main__':
    pop_size = 10
    population = generate_initial_population(pop_size)
    target = [7, 3, 2]

    fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

    num_generations = 50
    num_parents = 4

    for generation in range(num_generations):
        parents = select_parents(population, fitness_scores, num_parents)
        new_population = []

        for i in range(len(population) - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = parents + new_population
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        print(f"Generation {generation + 1}: Best Fitness = {min(fitness_scores)}")

    best_index = np.argmin(fitness_scores)
    best_robot = population[best_index]
    print(f"Best Robot: {best_robot}, Fitness: {fitness_scores[best_index]}")

