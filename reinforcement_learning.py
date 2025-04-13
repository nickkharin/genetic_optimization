import logging
import json
import gym
from gym import spaces
import numpy as np
from manipulator import Manipulator7DOF
from utilities import generate_random_target_in_half_sphere

class ManipulatorEnv(gym.Env):
    """
    Среда для обучения манипулятора с использованием RL.

    - Параметр radius=3.0 по умолчанию.
    - distance -> -3.0 * normalized_distance
    - bonus -> +800
    - movement penalty -> -0.003
    - energy penalty -> -0.008
    - randomize_start=False по умолчанию.

    Добавлено:
    1) Логирование метрик (distance, reward, energy) в JSON-файл,
       если enable_logging=True.
    2) max_episode_steps (по умолчанию 1000),
       чтобы гарантированно завершать эпизод, даже если
       робот не достиг цели (distance < 0.05).
    """

    def __init__(
            self,
            link_lengths=None,
            target_position=None,
            randomize_start=False,
            radius=3.0,
            enable_logging=False,
            log_path="rl_history.json",
            max_episode_steps=1000
    ):
        super(ManipulatorEnv, self).__init__()

        # 7 звеньев по умолчанию
        self.link_lengths = link_lengths if link_lengths else [1.0] * 7
        # Создаём манипулятор
        self.robot = Manipulator7DOF(lengths=self.link_lengths)
        self.target = target_position
        # По умолчанию отключаем случайный старт
        self.randomize_start = randomize_start
        # Радиус полусферы
        self.radius = radius

        # Действие: ±0.05 рад на каждый сустав
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(7,), dtype=np.float32
        )
        # Наблюдение: 7 углов + distance + stability + energy = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Логи
        self.trajectory = []
        self.energy_log = []
        self.joint_angle_log = []

        # Эпизодическая статистика
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_step = 0

        # Лимит шагов в эпизоде
        self.max_episode_steps = max_episode_steps

        # Логирование в JSON
        self.enable_logging = enable_logging
        self.log_path = log_path
        if self.enable_logging:
            # список, куда будем складывать данные об эпизодах
            self.training_history = []

    def max_reach(self):
        return np.sum(self.link_lengths)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_count += 1
        self.episode_reward = 0.0
        self.episode_step = 0

        # Если цель не задана, генерируем случайную в полусфере радиуса self.radius
        if self.target is None:
            self.target = generate_random_target_in_half_sphere(self.radius)

        self.robot.reset()

        # Если включить True, манипулятор стартует из случайных углов
        if self.randomize_start:
            random_angles = np.random.uniform(-np.pi, np.pi, size=7)
            self.robot.set_joint_angles(random_angles)

        self.trajectory.clear()
        self.energy_log.clear()
        self.joint_angle_log.clear()
        self.joint_angle_log.append(self.robot.get_joint_angles().copy())

        return self.get_observation(), {}

    def step(self, action):
        self.episode_step += 1

        # Применяем экшн
        current_angles = np.array(self.robot.get_joint_angles(), dtype=np.float32)
        new_angles = current_angles + action
        self.robot.set_joint_angles(new_angles)

        # Считаем награду
        reward = self.calculate_reward(action)
        self.episode_reward += reward

        # Проверяем дистанцию
        terminated = self.is_done()

        # Учитываем лимит шагов (truncated)
        truncated = False
        if self.episode_step >= self.max_episode_steps:
            truncated = True

        # Логируем текущее положение
        self.trajectory.append(self.robot.forward_kinematics()[-1])
        self.energy_log.append(self.robot.energy_consumption())
        self.joint_angle_log.append(self.robot.get_joint_angles().copy())

        # Если эпизод закончился (либо done, либо truncated),
        # записываем статистику
        if terminated or truncated:
            distance = np.linalg.norm(
                self.robot.forward_kinematics()[-1] - self.target
            )
            logging.info(
                f"Episode {self.episode_count} finished. "
                f"Steps={self.episode_step}, "
                f"EpReward={self.episode_reward:.3f}, Distance={distance:.3f}, "
                f"Terminated={terminated}, Truncated={truncated}"
            )

            if self.enable_logging:
                avg_energy = float(np.mean(self.energy_log)) if len(self.energy_log) > 0 else 0.0
                episode_data = {
                    'episode': self.episode_count,
                    'steps': self.episode_step,
                    'final_distance': float(distance),
                    'episode_reward': float(self.episode_reward),
                    'avg_energy': avg_energy,
                    'terminated': terminated,
                    'truncated': truncated
                }
                self.training_history.append(episode_data)

                # Записываем всё
                with open(self.log_path, 'w') as f:
                    json.dump(self.training_history, f, indent=2)

        return self.get_observation(), reward, (terminated or truncated), truncated, {"target": self.target}

    def get_observation(self):
        distance = np.linalg.norm(
            self.robot.forward_kinematics()[-1] - self.target
        )
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()
        return np.concatenate([
            self.robot.get_joint_angles(),
            [distance, stability, energy]
        ])

    def calculate_reward(self, action):
        current_ee_pos = self.robot.forward_kinematics()[-1]
        distance = np.linalg.norm(current_ee_pos - self.target)
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()
        joint_deltas = np.abs(action)

        # Нормализуем расстояние для штрафа
        normalized_distance = distance / (self.radius + 1e-8)
        normalized_energy = energy / 10.0
        trajectory_penalty = np.sum(joint_deltas)

        # Базовый штраф за дистанцию
        reward = (
            -3.0 * normalized_distance
            + 0.3 * stability
            - 0.008 * normalized_energy
            - 0.003 * trajectory_penalty
        )

        # Большой бонус за попадание
        if distance < 0.05:
            reward += 800.0

        return reward

    def is_done(self):
        distance = np.linalg.norm(
            self.robot.forward_kinematics()[-1] - self.target
        )
        return distance < 0.05

    def get_joint_angle_log(self):
        return np.array(self.joint_angle_log)