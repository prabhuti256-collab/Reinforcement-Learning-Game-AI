# Reinforcement Learning for Games Using Q-Learning

## Project Overview
This project implements a Reinforcement Learning agent that learns to solve a maze game using the Q-Learning algorithm. The agent interacts with the environment, receives rewards or penalties, and improves its decision-making through repeated training episodes.

## Objectives
- Implement a Q-Learning algorithm in Python.
- Train an AI agent to navigate a maze.
- Apply rewards and penalties to guide learning.
- Visualize training rewards and success rates.
- Test the trained agent in a graphical game environment.

## Technologies Used
- Python
- NumPy
- Pygame
- Matplotlib

## Algorithm
The project uses Q-Learning, a model-free reinforcement learning algorithm. The agent selects actions using an epsilon-greedy policy and updates Q-values based on rewards and estimated future values.

## Reward System
- Reaching the goal: +100
- Normal movement: -1
- Hitting a wall or boundary: -5

## Project Structure
- `environment.py` — Maze environment and reward system
- `agent.py` — Q-Learning implementation
- `train.py` — Agent training
- `play.py` — Game visualization and testing
- `visualize.py` — Training performance graphs
- `models/q_table.npy` — Saved Q-table
- `results/` — Training history and graphs

## How to Run
1. Install Python 3.12 or a compatible Python version.
2. Install dependencies using `pip install -r requirements.txt`.
3. Train the agent using `python train.py`.
4. Generate graphs using `python visualize.py`.
5. Play the game using `python play.py`.

## Training
The agent is trained for 5,000 episodes. During training, the program records rewards and whether the goal was reached.

## Results
The initial training run reported a 100% success rate over the final 100 episodes at each logged checkpoint. The final reported average reward was 86.22. These are training results; independent evaluation should be performed using the saved model.

## Future Improvements
- Implement Deep Q-Networks (DQN).
- Add different maze levels.
- Compare Q-Learning and DQN.
- Add a dashboard for training performance.

## Game Screenshot

![Maze Game](screenshots/maze_game.png)
