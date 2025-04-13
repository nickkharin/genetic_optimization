import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    # Загружаем оптимальные длины
    try:
        with open("optimal_lengths.json", "r") as f:
            optimal_lengths = json.load(f)
        logging.info(f"Оптимальные длины звеньев: {optimal_lengths}")
    except FileNotFoundError:
        logging.error("optimal_lengths.json не найден.")
        raise

    # Создаём среду (randomize_start=False) с радиусом R
    env = ManipulatorEnv(link_lengths=optimal_lengths, randomize_start=False, radius=R)

    # Загружаем обученную модель PPO
    model = PPO.load("ppo_manipulator")
    logging.info("Модель успешно загружена.")

    # Массивы для логирования траектории
    trajectory = []
    distances = []
    rewards = []
    energy_log = []  # чтобы записывать энергию покадрово

    # Дополнительно: лог для каждого шага
    test_steps_log = []

    # Сбрасываем среду
    obs, _ = env.reset()

    # 1) Считаем суммарную длину звеньев
    sum_len = sum(optimal_lengths)
    # 2) Точка (0, 0, sum_len), предполагаем «вертикаль»
    vertical_target = np.array([0.0, 0.0, sum_len])

    try:
        # Если в manipulator.py есть IK:
        best_angles = env.robot.inverse_kinematics_multiple_solutions(vertical_target)
        env.robot.set_joint_angles(best_angles)
        logging.info(f"Установлены углы IK для вертикали: {best_angles}")
    except Exception as e:
        logging.warning(f"Не удалось найти IK для (0,0,{sum_len}): {e}")
        # fallback — вручную зададим набор углов
        vertical_angles = [0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0]
        env.robot.set_joint_angles(vertical_angles)
        logging.info(f"Fallback: задали вручную углы = {vertical_angles}")

    # Записываем стартовую точку (после установки вертикали)
    start_position = env.robot.forward_kinematics()[-1]
    trajectory.append(start_position)
    start_distance = np.linalg.norm(start_position - env.target)
    distances.append(start_distance)
    rewards.append(0.0)
    energy_log.append(env.robot.energy_consumption())

    # Также сохраним начальное состояние
    test_steps_log.append({
        'step': 0,
        'distance': float(start_distance),
        'reward': 0.0,
        'energy': float(energy_log[-1])
    })

    done = False
    total_reward = 0
    step = 0
    max_steps = 1500  # более короткий лимит, например

    # Логика для «нет улучшений»
    no_improvement_count = 0
    best_distance_so_far = start_distance

    while not done and step < max_steps:
        # Если хотим случайный элемент (вдруг он выйдет из цикла):
        action, _ = model.predict(obs, deterministic=False)
        # Если хотим точно следовать политике:
        # action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        current_position = env.robot.forward_kinematics()[-1]
        dist = np.linalg.norm(current_position - env.target)
        current_energy = env.robot.energy_consumption()

        logging.info(f"Step {step}: Reward={reward:.3f}, Distance={dist:.3f}")

        trajectory.append(current_position)
        distances.append(dist)
        rewards.append(reward)
        energy_log.append(current_energy)

        # Лог для JSON
        test_steps_log.append({
            'step': step,
            'distance': float(dist),
            'reward': float(reward),
            'energy': float(current_energy)
        })

        # Проверяем улучшение
        if dist < best_distance_so_far - 0.001:
            best_distance_so_far = dist
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Если слишком долго нет улучшения, выходим
        if no_improvement_count > 300:
            logging.info(f"Нет улучшений {no_improvement_count} шагов подряд, прерываем...")
            break

        if done:
            logging.info(f"Манипулятор достиг цели на шаге {step}.")
            break

    # Финальный лог о результате
    if done and step < max_steps:
        logging.info(f"Тест завершён: манипулятор достиг цели. Итоговая награда: {total_reward:.3f}, Шаги: {step}")
    else:
        logging.info(f"Тест завершён: лимит шагов или отсутствие улучшений. Итоговая награда: {total_reward:.3f}, Шаги: {step}")

    # Сохраняем логи шага в JSON (test_log.json)
    with open("test_log.json", "w", encoding='utf-8') as f:
        json.dump(test_steps_log, f, indent=2)

    # Рассчитаем «среднюю энергию» и «среднюю дистанцию» за весь тест (если хотим)
    avg_energy = float(np.mean(energy_log))
    avg_dist = float(np.mean(distances))
    logging.info(f"Final test stats: avg_energy={avg_energy:.3f}, avg_distance={avg_dist:.3f}")

    # Визуализация
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Траектория
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        label='Trajectory',
        color='blue',
        marker='o'
    )

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color='yellow',
        label='Start Point',
        s=100
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
        color='green',
        label='End Point',
        s=100
    )
    ax.scatter(
        env.target[0], env.target[1], env.target[2],
        color='red',
        label='Target',
        s=100
    )

    # Отрисовка манипулятора (последняя конфигурация)
    def plot_manipulator(ax, robot, base_position=[0, 0, 0]):
        all_positions = robot.forward_kinematics()
        x_coords = [base_position[0]]
        y_coords = [base_position[1]]
        z_coords = [base_position[2]]
        for pos in all_positions:
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            z_coords.append(pos[2])
        ax.plot(x_coords, y_coords, z_coords, '-o', color='gray', label='Manipulator')
        ax.scatter(x_coords, y_coords, z_coords, color='black')
        return ax

    plot_manipulator(ax, env.robot, [0, 0, 0])

    # **Новый код**: рисуем «пол» (z=0) как 2D-плоскость
    plane_size = 5.0  # можно менять
    xx, yy = np.meshgrid(
        np.linspace(-plane_size, plane_size, 2),
        np.linspace(-plane_size, plane_size, 2)
    )
    zz = np.zeros_like(xx)  # z=0
    ax.plot_surface(xx, yy, zz, color='brown', alpha=0.2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Manipulator End-Effector Trajectory (with floor)')
    ax.legend()
    plt.show()