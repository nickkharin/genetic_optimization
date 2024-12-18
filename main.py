from genetic import *
from optimizer import GeneticAlgorithmOptimizer, decode_action
from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO
import logging
import numpy as np

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Этап 1: Оптимизация длин звеньев ---
    logging.info("Этап 1: Оптимизация длин звеньев с использованием генетического алгоритма.")

    num_states = 50
    optimizer = GeneticAlgorithmOptimizer(num_states=num_states, actions=range(5))
    current_state = 0

    num_generations = 50
    num_parents = 4
    best_fitness = float('inf')
    num_links = 7  # Количество звеньев манипулятора

    # Генерация начальной популяции
    population = generate_initial_population(pop_size=50, num_links=num_links)
    target = [2, 2, 0.2]  # Трехмерная цель (X, Y, Z)

    for generation in range(num_generations):
        action = optimizer.choose_action(current_state)
        mutation_rate, pop_size = decode_action(action)

        # Оценка текущей популяции
        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]
        parents = select_parents(population, fitness_scores, num_parents)

        # Элитарность: сохраняем лучших родителей
        new_population = parents.copy()

        while len(new_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)

            # Проверяем корректность входных данных
            if not isinstance(parent1, Manipulator7DOF) or not isinstance(parent2, Manipulator7DOF):
                raise TypeError("Родители должны быть экземплярами Manipulator7DOF.")

            # Кроссовер
            child = crossover(parent1, parent2)

            # Проверяем, что кроссовер возвращает корректный объект
            if not isinstance(child, Manipulator7DOF):
                raise TypeError("Ошибка: crossover вернул объект, не являющийся Manipulator7DOF.")

            # Мутация
            child = mutate(child, mutation_rate)

            # Проверка после мутации
            if not isinstance(child, Manipulator7DOF):
                raise TypeError("Ошибка: mutate вернул объект, не являющийся Manipulator7DOF.")

            new_population.append(child)

        population = new_population

        fitness_scores = [multi_criteria_fitness(robot, target) for robot in population]
        new_best_fitness = min(fitness_scores)
        logging.info(f"Generation {generation + 1}: Best Fitness = {new_best_fitness}")

        if new_best_fitness < best_fitness:
            reward = best_fitness - new_best_fitness
            best_fitness = new_best_fitness
        else:
            reward = 0

        next_state = optimizer.get_next_state(current_state, reward)
        optimizer.update_q_table(current_state, action, reward, next_state)
        current_state = next_state

    # Сохранение лучших длин звеньев
    best_index = np.argmin(fitness_scores)
    best_robot = population[best_index]
    optimal_lengths = best_robot.lengths
    logging.info(f"Лучшие длины звеньев: {optimal_lengths}, Fitness: {fitness_scores[best_index]}")

    # --- Этап 2: Управление углами суставов ---
    logging.info("Этап 2: Обучение управления углами суставов с фиксированными длинами звеньев.")

    # Создание среды с фиксированными длинами звеньев
    env = ManipulatorEnv(link_lengths=optimal_lengths)

    # Проверка цели в 3D и настройка среды
    logging.info(f"Цель установлена в координатах (X: {target[0]}, Y: {target[1]}, Z: {target[2]})")

    # Обучение RL-агента
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    # Сохранение обученной модели
    model.save("ppo_manipulator")
    logging.info("Обучение завершено. Модель сохранена как 'ppo_manipulator'.")