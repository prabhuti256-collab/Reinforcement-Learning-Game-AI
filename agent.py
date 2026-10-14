
import random
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        rows=8,
        cols=8,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.02
    ):
        self.rows = rows
        self.cols = cols
        self.n_actions = 4

        # Q-table: state (row, col) and action
        self.q_table = np.zeros(
            (rows, cols, self.n_actions)
        )

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, training=True):
        """Choose an action using epsilon-greedy policy."""
        row, col = state

        # Explore: try a random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # Exploit: choose the action with highest Q-value
        return int(np.argmax(self.q_table[row, col]))

    def learn(
        self, state, action, reward, next_state, done
    ):
        """Update the Q-value using the Q-learning formula."""
        row, col = state
        next_row, next_col = next_state

        old_value = self.q_table[row, col, action]

        if done:
            future_value = 0
        else:
            future_value = np.max(
                self.q_table[next_row, next_col]
            )

        # Q-learning update rule
        new_value = old_value + self.learning_rate * (
            reward
            + self.discount_factor * future_value
            - old_value
        )

        self.q_table[row, col, action] = new_value

    def decay_epsilon(self):
        """Reduce exploration as training progresses."""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    def save(self, filepath="models/q_table.npy"):
        """Save the learned Q-table."""
        np.save(filepath, self.q_table)

    def load(self, filepath="models/q_table.npy"):
        """Load a previously saved Q-table."""
        self.q_table = np.load(filepath)

