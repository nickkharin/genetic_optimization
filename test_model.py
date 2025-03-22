import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # иногда нужно явно
from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO


def plot_manipulator(ax, robot, base_position=[0, 0, 0]):
    """
    Отрисовка манипулятора на основе реальной кинематики из `robot`.
    Предполагаем, что robot.forward_kinematics() возвращает список [(x,y,z), ..., (x_end,y_end,z_end)].
    """
    all_positions = robot.forward_kinematics()

    x_coords = [base_position[0]]
    y_coords = [base_position[1]]
    z_coords = [base_position[2]]

    for pos in all_positions:
        x_coords.append(pos[0])
        y_coords.append(pos[1])
        z_coords.append(pos[2])

    ax.plot(x_coords, y_coords, z_coords, '-o', color='gray', label='Manipulator')
    ax.scatter(x_coords, y_coords, z_coords, color='black')  # суставы
    return ax


if __name__ == '__main__':
    R = 3.0
    logging.basicConfig(level=logging.INFO)

    # Загрузка оптимальных длин звеньев
    try:
        with open("optimal_lengths.json", "r") as f:
            optimal_lengths = json.load(f)
        logging.info(f"Оптимальные длины звеньев успешно загружены: {optimal_lengths}")
    except FileNotFoundError:
        logging.error("Файл optimal_lengths.json не найден. Проверьте выполнение первого этапа.")
        raise

    # Создаём окружение для тестирования (радиус=R, randomize_start=False)
    env = ManipulatorEnv(link_lengths=optimal_lengths, randomize_start=False, radius=R)

    # Загружаем обученную модель
    model = PPO.load("ppo_manipulator")
    logging.info("Модель успешно загружена.")

    # Готовим лог
    trajectory = []
    distances = []
    rewards = []

    logging.info("Начало тестирования...")

    # Сбрасываем среду
    obs, _ = env.reset()

    # Сразу сохраняем начальную позицию эффектора (до каких-либо действий)
    start_position = env.robot.forward_kinematics()[-1]
    trajectory.append(start_position)
    start_distance = np.linalg.norm(start_position - env.target)
    distances.append(start_distance)
    rewards.append(0.0)  # Нулевая награда до первого шага

    done = False
    total_reward = 0
    step = 0
    max_steps = 3000

    while not done and step < max_steps:
        # Детерминированный режим
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        current_position = env.robot.forward_kinematics()[-1]
        distance = np.linalg.norm(current_position - env.target)
        logging.info(f"Step {step}: Reward={reward:.3f}, Distance={distance:.3f}")

        trajectory.append(current_position)
        distances.append(distance)
        rewards.append(reward)

    if done:
        logging.info(f"Тест завершён: манипулятор достиг цели. Итоговая награда: {total_reward:.3f}, Шаги: {step}")
    else:
        logging.info(f"Тест завершён: лимит шагов. Итоговая награда: {total_reward:.3f}, Шаги: {step}")

    # Визуализация 3D
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Путь конца эффектора
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        label='Trajectory',
        color='blue',
        marker='o'
    )

    # Начальная, конечная точка, случайная цель
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
        color='yellow', label='Start Point', s=100
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
        color='green', label='End Point', s=100
    )
    ax.scatter(
        env.target[0], env.target[1], env.target[2],
        color='red', label='Target', s=100
    )

    # Отрисовка манипулятора (последняя конфигурация)
    plot_manipulator(ax, env.robot, base_position=[0, 0, 0])

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Manipulator End-Effector Trajectory')
    ax.legend()
    plt.show()
