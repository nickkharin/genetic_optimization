import json
import matplotlib.pyplot as plt

def plot_rl_history(json_path="rl_history.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = [d["episode"] for d in data]
    final_dist = [d["final_distance"] for d in data]
    ep_reward = [d["episode_reward"] for d in data]
    avg_energy = [d["avg_energy"] for d in data]

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, ep_reward, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('RL: Episode Reward over episodes')
    plt.legend()
    plt.savefig('rl_reward.png', dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, final_dist, label='Final Distance')
    plt.plot(episodes, avg_energy, label='Avg Energy', linestyle='--')
    plt.xlabel('Episode')
    plt.title('RL: Distance & Energy per Episode')
    plt.legend()
    plt.savefig('rl_dist_energy.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_rl_history()