import json
import matplotlib.pyplot as plt

def plot_ga_history(json_path="ga_history.json"):
    # 1) Загрузка данных
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2) Извлекаем ключевые поля
    generations = [d["generation"] for d in data]
    best_fit = [d["best_fitness"] for d in data]
    avg_fit = [d["avg_fitness"] for d in data]

    # Если вы записываете 'avg_distance' и 'avg_energy', то:
    avg_dist = [d["avg_distance"] for d in data]
    avg_energy = [d["avg_energy"] for d in data]

    # 3) Строим график фитнеса
    plt.figure(figsize=(8, 5))
    plt.plot(generations, best_fit, label='Best Fitness')
    plt.plot(generations, avg_fit, label='Average Fitness', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm: Fitness over Generations')
    plt.legend()
    # Сохраним в PNG
    plt.savefig('ga_fitness.png', dpi=150)
    plt.show()

    # 4) Дополнительно можем построить графики distance/energy
    plt.figure(figsize=(8, 5))
    plt.plot(generations, avg_dist, label='Avg Distance')
    plt.plot(generations, avg_energy, label='Avg Energy', linestyle='--')
    plt.xlabel('Generation')
    plt.title('GA: Distance & Energy over Generations')
    plt.legend()
    plt.savefig('ga_distance_energy.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_ga_history()