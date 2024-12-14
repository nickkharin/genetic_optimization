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
    def __init__(self, link_lengths=None, target_position=None):
        super(ManipulatorEnv, self).__init__()
        self.link_lengths = link_lengths if link_lengths else [1] * 7
        self.robot = Manipulator7DOF(lengths=self.link_lengths)
        self.target = target_position

        # Пространство действий
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float32)

        # Пространство наблюдений
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
        if self.target is None:
            max_reach = self.max_reach()
            r = np.random.uniform(0, 0.8 * max_reach)  # Цель ограничена 80% от максимальной дальности
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            self.target = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])

        # Устанавливаем манипулятор в начальное состояние
        self.robot.joint_angles = np.zeros(7)  # Углы суставов по умолчанию
        self.robot.reset()
        return self.get_observation(), {}

    def step(self, action):
        """
        Применяет действие и возвращает новое состояние.
        """
        self.robot.joint_angles += action
        reward = self.calculate_reward()
        terminated = self.is_done()
        info = {"target": self.target}
        return self.get_observation(), reward, terminated, False, info

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

        normalized_distance = distance / self.max_reach()
        normalized_stability = stability
        normalized_energy = energy / 10.0

        reward = -np.exp(normalized_distance) + 0.5 * normalized_stability - 0.1 * np.exp(normalized_energy)
        return reward

    def is_done(self):
        """
        Проверяет, достиг ли манипулятор цели.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        return distance < 0.05
