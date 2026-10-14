
import os
import numpy as np
import matplotlib.pyplot as plt

# Load training history
rewards = np.load("results/rewards.npy")
success = np.load("results/success.npy")

episodes = np.arange(1, len(rewards) + 1)
window = 100

os.makedirs("results", exist_ok=True)

# Graph 1: Rewards during training
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, alpha=0.25, label="Episode reward")

if len(rewards) >= window:
    moving_avg = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )
    plt.plot(
        np.arange(window, len(rewards) + 1),
        moving_avg,
        label="100-episode moving average"
    )

plt.title("Q-Learning Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/training_rewards.png")
plt.show()

# Graph 2: Success rate
if len(success) >= window:
    success_rate = np.convolve(
        success,
        np.ones(window) / window,
        mode="valid"
    ) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(window, len(success) + 1),
        success_rate
    )
    plt.title("AI Success Rate During Training")
    plt.xlabel("Episode")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/success_rate.png")
    plt.show()

print("Graphs saved in the results folder.")
print(f"Total episodes: {len(rewards)}")
print(f"Final 100-episode average reward: {np.mean(rewards[-100:]):.2f}")
print(f"Final 100-episode success rate: {np.mean(success[-100:]) * 100:.1f}%")

