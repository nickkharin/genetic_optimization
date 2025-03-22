import logging
import json

from genetic import genetic_algorithm, Manipulator7DOF
from reinforcement_learning import ManipulatorEnv
from stable_baselines3 import PPO

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Этап 1: Оптимизация длин звеньев (GA) ---
    logging.info("Этап 1: Оптимизация длин звеньев (GA) по нескольким точкам в полусфере.")

    # Зафиксированные параметры GA
    num_generations = 150
    pop_size = 100
    mutation_rate = 0.1  # Фиксированная скорость мутации
    num_links = 7

    # Параметры «многоцелевого» фитнеса
    n_samples = 5   # 5 случайных целей на каждую оценку
    R = 3.0     # радиус полусферы

    # Запуск GA (новая функция с «глобальной элитностью» внутри)
    best_robot = genetic_algorithm(
        pop_size=80,
        num_generations=100,
        mutation_rate=0.1,
        num_links=7,
        n_samples=5,
        max_r=R
    )

    # Извлекаем лучшие длины звеньев
    optimal_lengths = best_robot.lengths
    logging.info(f"Лучшие длины звеньев (GA): {optimal_lengths}")

    # Сохраняем их в JSON
    try:
        with open("optimal_lengths.json", "w") as f:
            json.dump(optimal_lengths, f)
        logging.info("Лучшие длины звеньев успешно сохранены в 'optimal_lengths.json'.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении длин звеньев: {e}")

    # --- Этап 2: Обучение RL (PPO) ---
    logging.info("Этап 2: Обучение RL (PPO) с найденной конфигурацией звеньев.")

    # Создаём среду с фиксированными длинами звеньев
    env = ManipulatorEnv(
        link_lengths=optimal_lengths,
        randomize_start=False,
        radius=R
    )

    # Обучение PPO
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
    model.learn(total_timesteps=300000)
    model.save("ppo_manipulator")

    logging.info("Обучение RL завершено. Модель сохранена как 'ppo_manipulator'.")