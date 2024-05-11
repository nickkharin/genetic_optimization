import numpy as np
import random


def decode_action(action):

    # Примерные значения, могут быть изменены на основе ваших требований
    mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.2]  # Разные скорости мутации
    population_sizes = [10, 20, 30, 40, 50]  # Разные размеры популяции
    mutation_rate = mutation_rates[action % len(mutation_rates)]
    pop_size = population_sizes[action % len(population_sizes)]
    return mutation_rate, pop_size


class GeneticAlgorithmOptimizer:
    def __init__(self, num_states, actions, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.q_table = np.zeros((num_states, len(actions)))  # Инициализация Q-таблицы
        self.alpha = alpha      # Коэффициент обучения
        self.gamma = gamma      # Фактор дисконтирования
        self.epsilon = epsilon  # Вероятность исследования
        self.actions = actions  # Возможные действия
        self.num_states = num_states

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def get_next_state(self, current_state, reward):
        # Пример: если было улучшение, перейдем к следующему состоянию
        if reward > 0:
            return min(current_state + 1, self.num_states - 1)
        return current_state  # Если улучшения нет, остаемся в текущем состоянии



