from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO
import logging
import numpy as np

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Заданные длины звеньев из этапа оптимизации
    optimal_lengths = [1.6427186647565832, 1.953847200857151, 1.547593575488852,
                       0.5253090580566936, 1.5216762501198067, 1.2436702018203596, 2.0326332247350996]

    # Задать тестовую цель для проверки
    target = np.array([2, 2, 0.2])

    # Создать окружение для тестирования
    env = ManipulatorEnv(link_lengths=optimal_lengths, target_position=target)

    # Загрузить обученную модель
    model = PPO.load("ppo_manipulator")
    logging.info("Модель успешно загружена.")

    # Запуск тестирования
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    logging.info("Начало тестирования...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Использовать детерминированные действия
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Логирование состояния и вознаграждения
        logging.info(f"Step {step}: Reward = {reward}, Observation = {obs}")

    logging.info(f"Тест завершён. Итоговая награда: {total_reward}, Количество шагов: {step}")
