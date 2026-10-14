from environment import MazeEnvironment

env = MazeEnvironment()

state = env.reset()
print("Starting position:", state)

state, reward, done = env.step(3)  # Move right

print("Next position:", state)
print("Reward:", reward)
print("Goal reached:", done)
