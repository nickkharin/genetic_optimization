import numpy as np
import random
from manipulator import Manipulator7DOF


def crossover(parent1, parent2):
    """
    Одноточечный кроссовер между двумя родителями.
    """
    if not isinstance(parent1, Manipulator7DOF) or not isinstance(parent2, Manipulator7DOF):
        raise TypeError("Родители должны быть объектами Manipulator7DOF.")

    crossover_point = random.randint(1, len(parent1.lengths) - 1)

    # Создание нового потомка с использованием длин звеньев родителей
    child_lengths = parent1.lengths[:crossover_point] + parent2.lengths[crossover_point:]
    child = Manipulator7DOF(lengths=child_lengths)

    return child


def mutate(robot, mutation_rate=0.1):
    """
    Мутация длин звеньев особи с заданной скоростью мутации.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    for i in range(len(robot.lengths)):
        if random.random() < mutation_rate:
            robot.lengths[i] += random.uniform(-0.1, 0.1)
            robot.lengths[i] = max(0.1, robot.lengths[i])  # Минимальная длина звена 0.1
    return robot


def select_parents(population, fitness_scores, num_parents):
    """
    Выбор родителей с использованием пропорционального отбора (рулетка).
    """
    fitness_scores = np.array(fitness_scores)
    if fitness_scores.sum() == 0:
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = fitness_scores / fitness_scores.sum()

    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return [population[i] for i in selected_indices]


def generate_initial_population(pop_size, num_links):
    """
    Генерация начальной популяции заданного размера с случайными длинами звеньев.
    """
    population = []
    for _ in range(pop_size):
        lengths = [random.uniform(0.5, 2.0) for _ in range(num_links)]
        robot = Manipulator7DOF(lengths=lengths)
        population.append(robot)
    return population


def multi_criteria_fitness(robot, target):
    """
    Оценка приспособленности особи на основе нескольких критериев.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    reachability_score = robot.max_reach()
    end_effector_position = robot.forward_kinematics()[-1]
    distance_to_target = np.linalg.norm(np.array(end_effector_position) - np.array(target))
    distance_score = -distance_to_target
    stability_score = robot.evaluate_stability()
    energy_score = -robot.energy_consumption()

    return reachability_score + 0.5 * distance_score + 0.3 * stability_score + 0.2 * energy_score


def genetic_algorithm(target, pop_size, num_generations, mutation_rate, num_links):
    """
    Основной цикл генетического алгоритма.
    """
    population = generate_initial_population(pop_size, num_links)

    for generation in range(num_generations):
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        best_fitness = max(fitness_scores)
        best_robot = population[np.argmax(fitness_scores)]
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

        parents = select_parents(population, fitness_scores, num_parents=pop_size // 2)

        new_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = crossover(parents[i], parents[i + 1])
                new_population.extend([child1, child2])

        # Обновленная логика мутации
        new_population = [
            mutate(robot, mutation_rate) if isinstance(robot, Manipulator7DOF) else robot
            for robot in new_population
        ]

        population = new_population

    fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]
    best_robot = population[np.argmax(fitness_scores)]
    return best_robot