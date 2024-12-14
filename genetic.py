import numpy as np
import random
from manipulator import Manipulator7DOF


def crossover(parent1, parent2):
    """
    Одноточечный кроссовер между двумя родителями.
    Возвращает двух потомков.
    """
    if not isinstance(parent1, Manipulator7DOF) or not isinstance(parent2, Manipulator7DOF):
        raise TypeError("Родители должны быть объектами Manipulator7DOF.")

    crossover_point = random.randint(1, len(parent1.lengths) - 1)

    # Создание двух потомков
    child1_lengths = parent1.lengths[:crossover_point] + parent2.lengths[crossover_point:]
    child2_lengths = parent2.lengths[:crossover_point] + parent1.lengths[crossover_point:]
    try:
        child1 = Manipulator7DOF(lengths=child1_lengths)
        child2 = Manipulator7DOF(lengths=child2_lengths)
    except Exception as e:
        raise ValueError(f"Ошибка создания потомка: {e}")

    return child1, child2


def mutate(robot, mutation_rate=0.1, min_length=0.5, max_length=2.0):
    """
    Мутация длины звеньев особи с контролем диапазона.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    for i in range(len(robot.lengths)):
        if random.random() < mutation_rate:
            mutation_value = random.uniform(-0.1, 0.1)
            robot.lengths[i] = max(min_length, min(max_length, robot.lengths[i] + mutation_value))
    return robot


def select_parents(population, fitness_scores, num_parents):
    """
    Выбор родителей с пропорциональным отбором (рулетка) и контролем на уникальность.
    """
    fitness_scores = np.array(fitness_scores)
    if fitness_scores.sum() == 0:
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = fitness_scores / fitness_scores.sum()

    selected_indices = set()
    while len(selected_indices) < num_parents:
        idx = np.random.choice(len(population), p=probabilities)
        selected_indices.add(idx)

    return [population[i] for i in selected_indices]


def generate_initial_population(pop_size, num_links, min_length=0.5, max_length=2.0):
    """
    Генерация начальной популяции с контролем диапазона длин звеньев.
    """
    population = []
    for _ in range(pop_size):
        lengths = [random.uniform(min_length, max_length) for _ in range(num_links)]
        robot = Manipulator7DOF(lengths=lengths)
        population.append(robot)
    return population


def multi_criteria_fitness(robot, target):
    """
    Оценка пригодности с нормализацией критериев.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    try:
        reachability_score = robot.max_reach() / 10.0  # Приведение к масштабу 0-1
        end_effector_position = robot.forward_kinematics()[-1]
        distance_to_target = np.linalg.norm(np.array(end_effector_position) - np.array(target))
        distance_score = -distance_to_target / 10.0  # Нормализация
        stability_score = robot.evaluate_stability() / 5.0  # Нормализация
        energy_score = -robot.energy_consumption() / 10.0  # Нормализация

        return reachability_score + 0.5 * distance_score + 0.3 * stability_score + 0.2 * energy_score
    except Exception as e:
        raise ValueError(f"Ошибка оценки пригодности: {e}")


def genetic_algorithm(target, pop_size, num_generations, mutation_rate, num_links):
    """
    Основной цикл генетического алгоритма.
    """
    # Генерация начальной популяции
    population = generate_initial_population(pop_size, num_links)

    for generation in range(num_generations):
        # Оценка пригодности
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        best_fitness = max(fitness_scores)
        best_robot = population[np.argmax(fitness_scores)]
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

        # Выбор родителей
        parents = select_parents(population, fitness_scores, num_parents=pop_size // 2)

        # Создание новой популяции через кроссовер
        new_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                try:
                    child1, child2 = crossover(parents[i], parents[i + 1])
                    new_population.extend([child1, child2])
                except ValueError as e:
                    print(f"Ошибка при кроссовере: {e}")

        # Мутация новой популяции
        new_population = [mutate(robot, mutation_rate) for robot in new_population]

        population = new_population

    # Финальная оценка
    fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]
    best_robot = population[np.argmax(fitness_scores)]
    return best_robot