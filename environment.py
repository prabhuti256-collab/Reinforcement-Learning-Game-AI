
import numpy as np

class MazeEnvironment:
    def __init__(self):
        self.rows = 8
        self.cols = 8

        self.start = (0, 0)
        self.goal = (7, 7)

        self.walls = {
            (0, 2), (1, 2), (2, 2),
            (2, 4), (3, 4), (4, 1),
            (4, 2), (4, 4), (5, 4),
            (6, 1), (6, 2)
        }

        # Up, Down, Left, Right
        self.actions = [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]

        self.state = self.start

    def reset(self):
        """Reset the agent to the starting position."""
        self.state = self.start
        return self.state

    def step(self, action):
        """Execute an action and return next state, reward, done."""
        row, col = self.state
        dr, dc = self.actions[action]

        next_state = (row + dr, col + dc)
        nr, nc = next_state

        # Check boundaries and walls
        if (
            nr < 0 or nr >= self.rows
            or nc < 0 or nc >= self.cols
            or next_state in self.walls
        ):
            return self.state, -5, False

        self.state = next_state

        # Goal reached
        if self.state == self.goal:
            return self.state, 100, True

        # Normal movement
        return self.state, -1, False

