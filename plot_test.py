import json
import matplotlib.pyplot as plt

def plot_test_log(json_path="test_log.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    steps = [d["step"] for d in data]
    distances = [d["distance"] for d in data]
    rewards = [d["reward"] for d in data]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, distances, label='Distance')
    plt.xlabel('Step')
    plt.ylabel('Distance to Target')
    plt.title('Test Scenario: Distance vs Step')
    plt.legend()
    plt.savefig('test_distance.png', dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, color='orange', label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Test Scenario: Reward vs Step')
    plt.legend()
    plt.savefig('test_reward.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_test_log()