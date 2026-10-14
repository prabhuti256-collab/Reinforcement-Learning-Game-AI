
import os
import numpy as np

from environment import MazeEnvironment
from agent import QLearningAgent


EPISODES = 5000
MAX_STEPS = 200


def train():
    env = MazeEnvironment()
    agent = QLearningAgent()

    rewards_history = []
    success_history = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        reached_goal = False

        for step in range(MAX_STEPS):
            # Select an action
            action = agent.choose_action(state)

            # Perform the action
            next_state, reward, done = env.step(action)

            # Learn from the experience
            agent.learn(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            total_reward += reward

            if done:
                reached_goal = True
                break

        # Reduce exploration gradually
        agent.decay_epsilon()

        rewards_history.append(total_reward)
        success_history.append(int(reached_goal))

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = (
                np.mean(success_history[-100:]) * 100
            )

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Average reward: {avg_reward:.2f} | "
                f"Success rate: {success_rate:.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save trained model and training history
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    agent.save("models/q_table.npy")

    np.save(
        "results/rewards.npy",
        np.array(rewards_history)
    )

    np.save(
        "results/success.npy",
        np.array(success_history)
    )

    print("\nTraining completed!")
    print("Model saved: models/q_table.npy")
    print("Rewards saved: results/rewards.npy")
    print("Success history saved: results/success.npy")

    return agent, rewards_history, success_history


if __name__ == "__main__":
    train()

