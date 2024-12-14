import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для 3D графиков
from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Загрузка оптимальных длин звеньев из файла JSON
    try:
        with open("optimal_lengths.json", "r") as f:
            optimal_lengths = json.load(f)
        logging.info(f"Оптимальные длины звеньев успешно загружены: {optimal_lengths}")
    except FileNotFoundError:
        logging.error("Файл optimal_lengths.json не найден. Проверьте выполнение первого этапа.")
        raise

    # Задать тестовую цель для проверки (с учетом координаты z)
    target = np.array([2, 2, 0.2])

    # Создать окружение для тестирования
    env = ManipulatorEnv(link_lengths=optimal_lengths, target_position=target)

    # Загрузить обученную модель
    model = PPO.load("ppo_manipulator")
    logging.info("Модель успешно загружена.")

    # Подготовка данных для визуализации
    trajectory = []  # Список для хранения траектории конца эффектора
    distances = []  # Список для расстояний до цели
    rewards = []  # Список для вознаграждений

    # Запуск тестирования
    obs, _ = env.reset()  # Извлекаем только obs из (obs, info)
    done = False
    total_reward = 0
    step = 0

    logging.info("Начало тестирования...")
    max_steps = 1000  # Установить лимит шагов

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)  # Использовать детерминированные действия
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Расчёт текущей позиции конца эффектора и расстояния до цели
        current_position = env.robot.forward_kinematics()[-1]
        distance = np.linalg.norm(current_position - target)

        # Логирование состояния
        logging.info(f"Step {step}: Reward = {reward}, Distance to Target = {distance}")

        # Сохранение данных для визуализации
        trajectory.append(current_position)
        distances.append(distance)
        rewards.append(reward)

    # Вывод итогов тестирования
    if done:
        logging.info(
            f"Тест завершён: манипулятор достиг цели. Итоговая награда: {total_reward}, Количество шагов: {step}")
    else:
        logging.info(
            f"Тест завершён: лимит шагов достигнут. Итоговая награда: {total_reward}, Количество шагов: {step}")

    # Построение 3D визуализации
    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Траектория движения конца эффектора
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory', color='blue', marker='o')

    # Начальная точка траектории
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='yellow', label='Start Point', s=100)

    # Финальная точка траектории
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='green', label='End Point', s=100)

    # Целевая точка
    ax.scatter(target[0], target[1], target[2], color='red', label='Target', s=100)

    # Подписи осей
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Заголовок
    ax.set_title('Manipulator End-Effector Trajectory')

    # Легенда
    ax.legend()

    # Показать 3D график
    plt.show()

    # Построение графика изменения расстояния до цели
    plt.figure(figsize=(10, 6))
    plt.plot(distances, label='Distance to Target', color='blue')
    plt.title('Distance to Target Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig("distance_visualization.png")
    plt.show()

    # Построение графика изменения вознаграждения
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Rewards', color='green')
    plt.title('Rewards Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("rewards_visualization.png")
    plt.show()