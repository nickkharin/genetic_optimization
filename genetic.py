import numpy as np
import random
from manipulator import Manipulator7DOF
from utilities import generate_random_target_in_half_sphere  # используем для многоцелевого фитнеса


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


def multi_criteria_fitness_single(robot, target):
    """
    Логика оценки для одной-единственной цели.
    """
    if not isinstance(robot, Manipulator7DOF):
        raise TypeError("Аргумент должен быть объектом Manipulator7DOF.")

    max_reach_val = robot.max_reach()
    end_effector_position = robot.forward_kinematics()[-1]
    distance_to_target = np.linalg.norm(np.array(end_effector_position) - np.array(target))

    # Если цель за пределами макс. досягаемости, штраф
    if distance_to_target > max_reach_val:
        return -5.0

    distance_score = -2.0 * distance_to_target  # чем меньше distance, тем выше фитнес
    stability_score = robot.evaluate_stability()
    energy_score = -robot.energy_consumption()  # больше энергии -> хуже
    reachability_bonus = max_reach_val * 0.2

    fitness = (
        distance_score
        + 0.3 * stability_score
        + 0.1 * energy_score
        + reachability_bonus
    )
    return fitness


def multi_criteria_fitness_multi(robot, n_samples=5, max_r=3.0):
    """
    Оцениваем фитнес робота по нескольким СЛУЧАЙНЫМ точкам в верхней полусфере радиуса max_r
    и берём среднее.
    """
    total_fitness = 0.0
    for _ in range(n_samples):
        random_target = generate_random_target_in_half_sphere(max_r)
        f_single = multi_criteria_fitness_single(robot, random_target)
        total_fitness += f_single
    return total_fitness / n_samples


def genetic_algorithm(pop_size, num_generations, mutation_rate, num_links,
                      n_samples=5, max_r=3.0):
    """
    Основной цикл генетического алгоритма.
    Оцениваем каждого робота на n_samples целей в полусфере радиуса max_r.
    Глобальная элитность: сохраняем лучшую особь за все поколения.

    + Защита от ситуации, когда select_parents вернёт очень мало родителей (избегаем бесконечного цикла).
    """
    population = generate_initial_population(pop_size, num_links)

    # Глобальные переменные для элитности
    global_best_fitness = float('-inf')
    global_best_robot = None

    for generation in range(num_generations):
        fitness_scores = []
        for robot in population:
            score = multi_criteria_fitness_multi(robot, n_samples, max_r)
            fitness_scores.append(score)

        # Лучшая особь в ТЕКУЩЕМ поколении
        best_fitness = max(fitness_scores)
        best_robot = population[np.argmax(fitness_scores)]

        print(f"Generation {generation + 1}: Best fitness = {best_fitness:.3f}, "
              f"Best lengths = {best_robot.lengths}")

        # Сравниваем с глобальным максимумом
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
            # Берём случайных родителей
            p1, p2 = random.sample(parents, 2)
            try:
                child1, child2 = crossover(p1, p2)
            except ValueError as e:
                print(f"Ошибка при кроссовере: {e}")
                # fallback — пропуск итерации
                continue

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Можно обрезать, если вдруг перекрыли размер
        population = new_population[:pop_size]

    # Оценка финальной популяции (необязательно)
    fitness_scores = [multi_criteria_fitness_multi(robot, n_samples, max_r) for robot in population]
    last_gen_best_fitness = max(fitness_scores)
    last_gen_best_robot = population[np.argmax(fitness_scores)]
    print(f"Final generation best fitness = {last_gen_best_fitness:.3f}, "
          f"Best lengths = {last_gen_best_robot.lengths}")

    print(f"Global best fitness = {global_best_fitness:.3f}")
    if global_best_robot is not None:
        print(f"Global best lengths = {global_best_robot.lengths}")
    else:
        print("Global best robot was None — что-то пошло не так.")

    # Возвращаем глобально лучшую особь за все поколения
    return global_best_robot if global_best_robot is not None else last_gen_best_robot