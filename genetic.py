import numpy as np
import random
import json

from manipulator import Manipulator7DOF
from utilities import generate_random_target_in_half_sphere


def crossover(parent1, parent2):
    """
    Одноточечный кроссовер между двумя родителями.
    Возвращает двух потомков.
    """
    if not isinstance(parent1, Manipulator7DOF) or not isinstance(parent2, Manipulator7DOF):
        raise TypeError("Родители должны быть объектами Manipulator7DOF.")

    crossover_point = random.randint(1, len(parent1.lengths) - 1)

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
            new_length = robot.lengths[i] + mutation_value
            # Контроль в пределах min_length..max_length
            robot.lengths[i] = max(min_length, min(max_length, new_length))
    return robot


def select_parents(population, fitness_scores, num_parents):
    """
    Выбор родителей с пропорциональным отбором (рулетка) и контролем на уникальность.
    """
    fitness_scores = np.array(fitness_scores, dtype=np.float64)
    min_val = fitness_scores.min()
    if min_val < 0:
        # Сдвигаем все фитнесы к положительным значениям
        fitness_scores = fitness_scores - min_val + 1e-9

    total = fitness_scores.sum()
    if total == 0:
        # Если после сдвига всё равно 0, значит все были одинаковые
        probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
    else:
        probabilities = fitness_scores / total

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


def multi_criteria_fitness_single_details(robot, target):
    """
    Вычисляет fitness, а также возвращает фактическую дистанцию и энергию.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    max_reach_val = robot.max_reach()
    end_effector_position = robot.forward_kinematics()[-1]
    distance_to_target = np.linalg.norm(np.array(end_effector_position) - np.array(target))

    # Если цель за пределами макс. досягаемости, штраф
    if distance_to_target > max_reach_val:
        # fitness = -5, возвращаем distance и energy
        return -5.0, distance_to_target, robot.energy_consumption()

    distance_score = -2.0 * distance_to_target
    stability_score = robot.evaluate_stability()
    energy_now = robot.energy_consumption()
    energy_score = -energy_now  # больше энергии -> хуже
    reachability_bonus = max_reach_val * 0.2

    fitness = (
        distance_score
        + 0.3 * stability_score
        + 0.1 * energy_score
        + reachability_bonus
    )
    return fitness, distance_to_target, energy_now


def multi_criteria_fitness_multi_details(robot, n_samples=5, max_r=3.0):
    """
    Возвращает усреднённые (fitness, distance, energy) по нескольким точкам.
    """
    total_fitness = 0.0
    total_distance = 0.0
    total_energy = 0.0

    for _ in range(n_samples):
        random_target = generate_random_target_in_half_sphere(max_r)
        f_single, dist_single, energy_single = multi_criteria_fitness_single_details(robot, random_target)
        total_fitness += f_single
        total_distance += dist_single
        total_energy += energy_single

    avg_fitness = total_fitness / n_samples
    avg_distance = total_distance / n_samples
    avg_energy = total_energy / n_samples

    return avg_fitness, avg_distance, avg_energy


def genetic_algorithm(pop_size, num_generations, mutation_rate, num_links,
                      n_samples=5, max_r=3.0):
    """
    Основной цикл генетического алгоритма.
    Теперь мы используем функции, возвращающие также среднюю дистанцию и энергию,
    чтобы собрать более подробную статистику в history.
    """
    population = generate_initial_population(pop_size, num_links)

    # Глобальные переменные для элитности
    global_best_fitness = float('-inf')
    global_best_robot = None

    # Список для логирования динамики поколений
    history = []

    for generation in range(num_generations):
        fitness_scores = []
        distances_list = []
        energies_list = []

        # Оценка популяции
        for robot in population:
            # Используем детальную функцию
            avg_fit, avg_dist, avg_en = multi_criteria_fitness_multi_details(
                robot, n_samples, max_r
            )
            fitness_scores.append(avg_fit)
            distances_list.append(avg_dist)
            energies_list.append(avg_en)

        avg_fitness = float(np.mean(fitness_scores))
        best_fitness = float(max(fitness_scores))
        best_index = int(np.argmax(fitness_scores))
        best_robot = population[best_index]

        avg_distance = float(np.mean(distances_list))
        avg_energy = float(np.mean(energies_list))

        print(f"Generation {generation + 1}: best_fitness={best_fitness:.3f}, "
              f"avg_fitness={avg_fitness:.3f}, avg_dist={avg_distance:.3f}, avg_energy={avg_energy:.3f}")

        # Логируем текущие результаты
        history.append({
            'generation': generation + 1,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'avg_distance': avg_distance,
            'avg_energy': avg_energy
        })

        # Проверяем глобальный максимум
        if best_fitness > global_best_fitness:
            import copy
            global_best_fitness = best_fitness
            global_best_robot = copy.deepcopy(best_robot)

        # Селекция
        num_parents = pop_size // 2
        parents = select_parents(population, fitness_scores, num_parents)
        # Защита: если вдруг parents слишком мало (<2), fallback -> вся популяция
        if len(parents) < 2:
            parents = population[:]

        # Создаём новую популяцию
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.sample(parents, 2)
            try:
                child1, child2 = crossover(p1, p2)
            except ValueError as e:
                print(f"Ошибка при кроссовере: {e}")
                continue

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:pop_size]

    # Оценка финальной популяции
    final_fitness_scores = []
    final_distances = []
    final_energies = []

    for robot in population:
        fit_val, dist_val, en_val = multi_criteria_fitness_multi_details(
            robot, n_samples, max_r
        )
        final_fitness_scores.append(fit_val)
        final_distances.append(dist_val)
        final_energies.append(en_val)

    last_gen_best_fitness = max(final_fitness_scores)
    last_gen_best_robot = population[np.argmax(final_fitness_scores)]

    print(f"Final generation best fitness = {last_gen_best_fitness:.3f}, "
          f"Best lengths = {last_gen_best_robot.lengths}")

    print(f"Global best fitness = {global_best_fitness:.3f}")
    if global_best_robot is not None:
        print(f"Global best lengths = {global_best_robot.lengths}")
    else:
        print("Global best robot was None — что-то пошло не так.")

    # Сохраняем историю поколений в JSON
    with open("ga_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Возвращаем глобально лучшую особь за все поколения
    return global_best_robot if global_best_robot is not None else last_gen_best_robot