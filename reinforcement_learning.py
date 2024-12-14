import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from manipulator import Manipulator7DOF
import matplotlib.pyplot as plt


class ManipulatorEnv(gym.Env):
    """
    Среда для обучения манипулятора с использованием RL.
    """
    def __init__(self, link_lengths=[1, 1, 1, 1, 1, 1, 1], target_position=None):
        super(ManipulatorEnv, self).__init__()
        self.link_lengths = link_lengths  # Длины звеньев манипулятора
        self.robot = Manipulator7DOF(lengths=self.link_lengths)  # Передаём длины звеньев как lengths
        self.target = target_position  # Если цель передана, используем её, иначе сгенерируем в reset()

        # Пространство действий: изменения углов суставов
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float32)

        # Пространство наблюдений: 7 углов суставов + расстояние до цели + стабильность + энергопотребление
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def max_reach(self):
        """
        Рассчитывает максимальную дальность манипулятора.
        """
        return np.sum(self.link_lengths)

    def reset(self, seed=None, options=None):
        """
        Сбрасывает состояние среды и генерирует цель в пределах досягаемости манипулятора.
        """
        super().reset(seed=seed)
        max_reach = self.max_reach()

        if self.target is None:  # Если цель не задана, генерируем случайную
            r = np.random.uniform(0, max_reach)  # Радиус (не больше max_reach)
            theta = np.random.uniform(0, 2 * np.pi)  # Угол в горизонтальной плоскости
            phi = np.random.uniform(0, np.pi)  # Угол в вертикальной плоскости

            self.target = np.array([
                r * np.sin(phi) * np.cos(theta),  # X
                r * np.sin(phi) * np.sin(theta),  # Y
                r * np.cos(phi)                   # Z
            ])

        self.robot = Manipulator7DOF(lengths=self.link_lengths)
        return self.get_observation(), {}

    def seed(self, seed=None):
        """
        Устанавливает начальное состояние генератора случайных чисел.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Применяет действие и возвращает новое состояние, награду, флаги завершения и дополнительную информацию.
        """
        self.robot.joint_angles += action
        reward = self.calculate_reward()
        terminated = self.is_done()  # Завершение эпизода (например, достижение цели)
        truncated = False  # Пока не используется ограничение длины эпизода
        info = {"target": self.target}  # Дополнительная информация о текущей цели
        return self.get_observation(), reward, terminated, truncated, info

    def get_observation(self):
        """
        Возвращает текущее состояние среды.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()
        return np.concatenate([self.robot.joint_angles, [distance, stability, energy]])

    def calculate_reward(self):
        """
        Вычисляет награду на основе расстояния до цели, стабильности и энергопотребления.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()

        # Нормализуем значения
        max_distance = self.max_reach()
        normalized_distance = distance / max_distance
        normalized_stability = stability  # Если стабильность уже от 0 до 1
        normalized_energy = energy / 10.0  # Зависит от модели

        # Взвешенная функция награды
        reward = -normalized_distance + 0.5 * normalized_stability - 0.1 * normalized_energy
        return reward

    def is_done(self):
        """
        Проверяет, достиг ли манипулятор цели.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        return distance < 0.05
