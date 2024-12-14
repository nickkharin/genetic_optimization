import numpy as np
import random
from manipulator import Manipulator7DOF


def crossover(parent1, parent2):
    """
    Одноточечный кроссовер между двумя родителями.
    """
    # Генерируем точку кроссовера
    crossover_point = random.randint(1, len(parent1.lengths) - 1)

    # Создаём ребёнка, комбинируя длины звеньев родителей
    child_lengths = parent1.lengths[:crossover_point] + parent2.lengths[crossover_point:]

    # Создаём нового манипулятора с новыми длинами звеньев
    child = Manipulator7DOF(lengths=child_lengths)
    return child


def mutate(robot, mutation_rate=0.1):
    """
    Мутация длин звеньев особи с заданной скоростью мутации.
    """
    for i in range(len(robot.lengths)):
        if random.random() < mutation_rate:
            # Мутация длины звена
            robot.lengths[i] += random.uniform(-0.1, 0.1)
            robot.lengths[i] = max(0.1, robot.lengths[i])  # Минимальная длина звена 0.1
    return robot


def select_parents(population, fitness_scores, num_parents):
    """
    Выбор родителей с использованием пропорционального отбора (рулетка).
    """
    fitness_scores = np.array(fitness_scores)
    probabilities = fitness_scores / fitness_scores.sum()  # Вероятности пропорциональны фитнесу
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return [population[i] for i in selected_indices]


def generate_initial_population(pop_size, num_links):
    """
    Генерация начальной популяции заданного размера с случайными длинами звеньев.
    """
    population = []
    for _ in range(pop_size):
        # Случайные длины звеньев в диапазоне [0.5, 2.0]
        lengths = [random.uniform(0.5, 2.0) for _ in range(num_links)]
        robot = Manipulator7DOF(lengths=lengths)
        population.append(robot)
    return population


def multi_criteria_fitness(robot, target):
    """
    Оценка приспособленности особи на основе нескольких критериев.
    """
    # Расчёт досягаемости
    reachability_score = robot.max_reach()

    # Расстояние от цели
    distance_score = -np.linalg.norm(np.array(robot.forward_kinematics()[-1]) - np.array(target))

    # Стабильность
    stability_score = robot.evaluate_stability()

    # Энергопотребление
    energy_score = -robot.energy_consumption()

    # Итоговая фитнес-функция
    return reachability_score + 0.5 * distance_score + 0.3 * stability_score + 0.2 * energy_score


def genetic_algorithm(target, pop_size, num_generations, mutation_rate, num_links):
    """
    Основной цикл генетического алгоритма.
    """
    # Генерация начальной популяции
    population = generate_initial_population(pop_size, num_links)

    for generation in range(num_generations):
        # Оценка приспособленности каждой особи
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]

        # Логирование лучшей особи
        best_fitness = max(fitness_scores)
        best_robot = population[np.argmax(fitness_scores)]
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

        # Выбор родителей
        parents = select_parents(population, fitness_scores, num_parents=pop_size // 2)

        # Создание нового поколения через кроссовер и мутацию
        new_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1 = crossover(parents[i], parents[i + 1])
                child2 = crossover(parents[i + 1], parents[i])
                new_population.extend([child1, child2])

        # Применение мутации
        new_population = [mutate(robot, mutation_rate) for robot in new_population]

        # Замена старой популяции новой
        population = new_population

    # Возврат лучшей особи из последнего поколения
    fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]
    best_robot = population[np.argmax(fitness_scores)]
    return best_robot