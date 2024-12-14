import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from manipulator import Manipulator7DOF


class ManipulatorEnv(gym.Env):
    """
    Среда для обучения манипулятора с использованием RL.
    """
    def __init__(self, link_lengths=[1, 1, 1, 1, 1, 1, 1]):
        super(ManipulatorEnv, self).__init__()
        self.link_lengths = link_lengths  # Длины звеньев манипулятора
        self.robot = Manipulator7DOF(lengths=self.link_lengths)  # Исправлено: передаём как lengths
        self.target = None  # Цель будет генерироваться в reset()

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

        # Генерация случайной точки внутри сферы досягаемости
        r = np.random.uniform(0, max_reach)  # Радиус (не больше max_reach)
        theta = np.random.uniform(0, 2 * np.pi)  # Угол в горизонтальной плоскости
        phi = np.random.uniform(0, np.pi)  # Угол в вертикальной плоскости

        self.target = np.array([
            r * np.sin(phi) * np.cos(theta),  # X
            r * np.sin(phi) * np.sin(theta),  # Y
            r * np.cos(phi)                   # Z
        ])

        self.robot = Manipulator7DOF(lengths=self.link_lengths)  # Исправлено: передаём как lengths
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
        Вычисляет награду на основе расстояния до цели.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        stability = self.robot.evaluate_stability()
        return -distance + stability

    def is_done(self):
        """
        Проверяет, достиг ли манипулятор цели.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        return distance < 0.05


# Для интеграции с `main.py` ниже блок `if __name__ == "__main__"` лучше убрать,
# но если хотите запускать этот файл отдельно, можно оставить пример RL-обучения.

if __name__ == "__main__":
    # Длины звеньев манипулятора
    link_lengths = [1.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.3]

    # Создать векторизованную среду для обучения
    env = make_vec_env(ManipulatorEnv, n_envs=4, env_kwargs={"link_lengths": link_lengths})

    # Создать модель PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Обучить модель
    model.learn(total_timesteps=100000)

    # Сохранить модель
    model.save("ppo_manipulator")

    # Тестирование модели
    test_env = ManipulatorEnv(link_lengths=link_lengths)
    obs, _ = test_env.reset()  # Извлекаем только obs из (obs, info)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)  # Обрабатываем 5 значений
        print(f"Reward: {reward}, Observation: {obs}, Target: {info['target']}")