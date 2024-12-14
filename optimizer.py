import numpy as np
import random
import logging


def decode_action(action):
    """
    Декодирует действие в параметры генетического алгоритма.
    """
    mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    population_sizes = [10, 20, 30, 40, 50]
    mutation_index = action // len(population_sizes) % len(mutation_rates)
    population_index = action % len(population_sizes)
    mutation_rate = mutation_rates[mutation_index]
    pop_size = population_sizes[population_index]
    return mutation_rate, pop_size


class GeneticAlgorithmOptimizer:
    """
    Оптимизатор для генетического алгоритма с использованием Q-learning.
    """
    def __init__(self, num_states, actions, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.q_table = np.zeros((num_states, len(actions)))  # Инициализация Q-таблицы
        self.alpha = alpha      # Коэффициент обучения
        self.gamma = gamma      # Фактор дисконтирования
        self.epsilon = epsilon  # Вероятность исследования
        self.actions = actions  # Возможные действия
        self.num_states = num_states

    def choose_action(self, state):
        """
        Выбирает действие на основе Q-таблицы или случайно.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)  # Исследование пространства действий
        else:
            action = np.argmax(self.q_table[state])  # Использование знаний
        logging.info(f"State: {state}, Chosen action: {action}, Epsilon: {self.epsilon}")
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Обновляет Q-таблицу на основе текущего состояния, действия, награды и следующего состояния.
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = np.clip(new_value, -1e3, 1e3)  # Ограничение значений Q

    def get_next_state(self, current_state, reward):
        """
        Определяет следующее состояние на основе награды.
        """
        if reward > 0:
            return min(current_state + 1, self.num_states - 1)
        elif reward < 0:
            return max(current_state - 1, 0)
        return current_state