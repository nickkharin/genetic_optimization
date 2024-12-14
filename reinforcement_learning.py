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

        # Сохранение траекторий для анализа
        self.trajectory = []
        self.energy_log = []

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
            r = np.random.uniform(0, max_reach)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            self.target = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
        self.robot.reset()
        self.trajectory = []
        self.energy_log = []
        return self.get_observation(), {}

    def step(self, action):
        """
        Применяет действие и возвращает новое состояние.
        """
        self.robot.joint_angles += action
        reward = self.calculate_reward(action)
        terminated = self.is_done()
        info = {"target": self.target}
        self.trajectory.append(self.robot.forward_kinematics()[-1])
        self.energy_log.append(self.robot.energy_consumption())
        return self.get_observation(), reward, terminated, False, info

    def get_observation(self):
        """
        Возвращает текущее состояние среды.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()
        return np.concatenate([self.robot.joint_angles, [distance, stability, energy]])

    def calculate_reward(self, action):
        """
        Вычисляет награду на основе расстояния до цели, стабильности, энергопотребления и плавности.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        stability = self.robot.evaluate_stability()
        energy = self.robot.energy_consumption()
        joint_deltas = np.abs(action)

        normalized_distance = distance / self.max_reach()
        normalized_stability = stability
        normalized_energy = energy / 10.0
        trajectory_penalty = np.sum(joint_deltas)

        reward = (
            -normalized_distance  # Чем ближе к цели, тем лучше
            + 0.5 * normalized_stability  # Стабильность
            - 0.1 * normalized_energy  # Штраф за энергозатраты
            - 0.05 * trajectory_penalty  # Штраф за резкие движения
        )

        # Бонус за достижение цели
        if distance < 0.05:
            reward += 100
        return reward

    def is_done(self):
        """
        Проверяет, достиг ли манипулятор цели.
        """
        distance = np.linalg.norm(self.robot.forward_kinematics()[-1] - self.target)
        return distance < 0.05


if __name__ == "__main__":
    link_lengths = [1.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.3]

    env = make_vec_env(ManipulatorEnv, n_envs=4, env_kwargs={"link_lengths": link_lengths})

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    model.learn(total_timesteps=100000)
    model.save("ppo_manipulator")

    test_env = ManipulatorEnv(link_lengths=link_lengths, target_position=np.array([2.0, 2.0, 0.5]))
    obs, _ = test_env.reset()
    done = False

    rewards = []
    distances = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = test_env.step(action)
        rewards.append(reward)
        distances.append(np.linalg.norm(info['target'] - test_env.robot.forward_kinematics()[-1]))
        print(f"Reward: {reward}, Observation: {obs}, Target: {info['target']}")

    # Визуализация наград
    plt.figure()
    plt.plot(rewards, label="Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Reward vs Steps")
    plt.legend()
    plt.show()

    # Визуализация расстояния
    plt.figure()
    plt.plot(distances, label="Distance to Target")
    plt.xlabel("Steps")
    plt.ylabel("Distance")
    plt.title("Distance to Target vs Steps")
    plt.legend()
    plt.show()

    # Визуализация энергозатрат
    plt.figure()
    plt.plot(test_env.energy_log, label="Energy Consumption")
    plt.xlabel("Steps")
    plt.ylabel("Energy")
    plt.title("Energy Consumption vs Steps")
    plt.legend()
    plt.show()