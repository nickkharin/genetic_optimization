import numpy as np
import random
import logging

def decode_action(action):
    """
    Декодирует действие в параметры генетического алгоритма.

    Args:
        action (int): Код действия.

    Returns:
        tuple: Состоит из коэффициента мутации (mutation_rate) и размера популяции (pop_size).
    """
    mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    population_sizes = [10, 20, 30, 40, 50]
    mutation_index = (action // len(population_sizes)) % len(mutation_rates)
    population_index = action % len(population_sizes)
    mutation_rate = mutation_rates[mutation_index]
    pop_size = population_sizes[population_index]
    return mutation_rate, pop_size

class GeneticAlgorithmOptimizer:
    """
    Оптимизатор для генетического алгоритма с использованием Q-learning.

    Attributes:
        q_table (np.ndarray): Q-таблица для хранения значений состояний-действий.
        alpha (float): Коэффициент обучения (скорость обновления Q-значений).
        gamma (float): Фактор дисконтирования (учет будущих наград).
        epsilon (float): Вероятность случайного выбора действия (исследование).
        actions (list): Список доступных действий.
        num_states (int): Количество возможных состояний.
    """
    def __init__(self, num_states, actions, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.q_table = np.zeros((num_states, len(actions)))  # Инициализация Q-таблицы
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Фактор дисконтирования
        self.epsilon = epsilon  # Вероятность исследования
        self.actions = actions  # Доступные действия
        self.num_states = num_states  # Количество состояний

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)  # Исследование пространства действий
            logging.info(f"Exploration: Random action chosen: {action}")
        else:
            action = np.argmax(self.q_table[state])  # Использование знаний
            logging.info(f"Exploitation: Best action chosen: {action}")
        logging.info(f"State: {state}, Action: {action}, Q-values: {self.q_table[state]}")
        return action

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = np.clip(new_value, -1e3, 1e3)  # Ограничение значений Q

        # Логирование изменений
        logging.info(f"Updated Q-table: State {state}, Action {action}, Old Value {old_value}, "
                     f"Reward {reward}, Next Max {next_max}, New Value {new_value}")

    def get_next_state(self, current_state, reward):
        """
        Определяет следующее состояние на основе текущего состояния и награды.

        Args:
            current_state (int): Текущее состояние.
            reward (float): Полученная награда.

        Returns:
            int: Следующее состояние.
        """
        if reward > 0:
            return min(current_state + 1, self.num_states - 1)
        elif reward < 0:
            return max(current_state - 1, 0)
        return current_state