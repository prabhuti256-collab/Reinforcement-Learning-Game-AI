
from agent import QLearningAgent

agent = QLearningAgent()

state = (0, 0)
action = agent.choose_action(state)

print("Chosen action:", action)
print("Q-table shape:", agent.q_table.shape)

next_state = (0, 1)
reward = -1
done = False

agent.learn(
    state, action, reward, next_state, done
)

print("Q-value updated successfully!")
print("Updated Q-value:", agent.q_table[0, 0, action])

