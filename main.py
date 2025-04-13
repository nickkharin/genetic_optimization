import logging
import json
import time
import numpy as np

from genetic import genetic_algorithm
from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO


def evaluate_configuration(link_lengths, n_steps=50, radius=3.0, model=None):
    """
    Короткий тестовый сценарий, имитирующий работу манипулятора:
      - Если 'model' указан (PPO), то используем детерминированный policy для step()
      - Иначе случайные действия.
    Возвращает средние метрики (энергия, distance).
    """
    env = ManipulatorEnv(
        link_lengths=link_lengths,
        randomize_start=False,
        radius=radius,
        enable_logging=False
    )
    obs, _ = env.reset()

    total_energy = 0.0
    total_distance = 0.0
    step_count = 0

    for step in range(n_steps):
        if model is not None:
            # Используем обученную policy
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Случайное действие
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)

        current_energy = env.robot.energy_consumption()
        total_energy += current_energy

        # Расстояние до цели:
        distance = np.linalg.norm(env.robot.forward_kinematics()[-1] - env.target)
        total_distance += distance

        step_count += 1

        if done:
            # Если цель достигнута, прерываем
            break

    avg_energy = total_energy / (step_count if step_count > 0 else 1)
    avg_distance = total_distance / (step_count if step_count > 0 else 1)

    return {
        'avg_energy': float(avg_energy),
        'avg_distance': float(avg_distance),
        'steps_completed': step_count
    }


def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Baseline (исходная) конфигурация
    # Допустим, все длины = 1.0:
    baseline_lengths = [1.0] * 7
    logging.info("### Evaluating baseline configuration ###")
    baseline_result = evaluate_configuration(
        link_lengths=baseline_lengths,
        n_steps=50,
        radius=3.0,
        model=None  # без RL policy
    )
    logging.info(f"Baseline result: {baseline_result}")

    # 2) Запускаем генетический алгоритм
    pop_size = 80
    num_generations = 80
    mutation_rate = 0.1
    num_links = 7
    n_samples = 5
    max_r = 3.0

    logging.info("### Running Genetic Algorithm ###")
    start_time = time.time()
    best_robot = genetic_algorithm(
        pop_size=pop_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        num_links=num_links,
        n_samples=n_samples,
        max_r=max_r
    )
    ga_time = time.time() - start_time
    logging.info(f"Genetic Algorithm took {ga_time:.2f} seconds.")

    # 3) Сохраняем лучшие длины звеньев
    optimal_lengths = best_robot.lengths
    with open("optimal_lengths.json", "w") as f:
        json.dump(optimal_lengths, f)
    logging.info(f"Optimal lengths saved to 'optimal_lengths.json': {optimal_lengths}")

    # 4) Обучаем RL (PPO) c уже «оптимизированными» длинами звеньев
    logging.info("### Training PPO with optimized configuration ###")
    env = ManipulatorEnv(
        link_lengths=optimal_lengths,
        randomize_start=False,
        radius=max_r,
        enable_logging=False
    )
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./ppo_logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    model.learn(total_timesteps=400000)
    model.save("ppo_manipulator")
    logging.info("RL model ppo_manipulator saved.")

    # 5) Evaluate optimized config (with trained PPO policy)
    logging.info("### Evaluating optimized configuration with RL policy ###")
    # reload model just in case
    trained_model = PPO.load("ppo_manipulator")
    optimized_result = evaluate_configuration(
        link_lengths=optimal_lengths,
        n_steps=50,
        radius=max_r,
        model=trained_model  # используем детерминированный policy PPO
    )
    logging.info(f"Optimized (with RL) result: {optimized_result}")

    # 6) Сравнение «baseline vs. optimized»
    compare_data = {
        'baseline': baseline_result,
        'optimized_RL': optimized_result
    }
    with open("baseline_vs_optimal.json", "w") as f:
        json.dump(compare_data, f, indent=2)
    logging.info("Comparison 'baseline_vs_optimal.json' written.")

    logging.info("### All steps done. ###")


if __name__ == '__main__':
    main()
