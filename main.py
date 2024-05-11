from genetic import *
from optimizer import GeneticAlgorithmOptimizer, decode_action

if __name__ == '__main__':
    num_states = 50  # Примерное количество состояний
    optimizer = GeneticAlgorithmOptimizer(num_states=num_states, actions=range(5))
    current_state = 0

    num_generations = 50
    num_parents = 4
    best_fitness = float('inf')

    for generation in range(num_generations):
        action = optimizer.choose_action(current_state)
        mutation_rate, pop_size = decode_action(action)  # Декодирование действия

        population = generate_initial_population(pop_size)
        target = [7, 3, 2]
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        for _ in range(len(population)):
            parents = select_parents(population, fitness_scores, num_parents)
            new_population = []

            for i in range(len(population) - len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_population.append(child)

            population = parents + new_population
            fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        print(f"Generation {generation + 1}: Best Fitness = {min(fitness_scores)}")

        new_best_fitness = min(fitness_scores)
        if new_best_fitness < best_fitness:
            reward = best_fitness - new_best_fitness
            best_fitness = new_best_fitness
        else:
            reward = 0

        next_state = optimizer.get_next_state(current_state, reward)  # Теперь передаем текущее состояние и вознаграждение
        optimizer.update_q_table(current_state, action, reward, next_state)
        current_state = next_state

    best_index = np.argmin(fitness_scores)
    best_robot = population[best_index]
    print(f"Best Robot: {best_robot}, Fitness: {fitness_scores[best_index]}")
