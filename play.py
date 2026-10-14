
import os
import pygame

from environment import MazeEnvironment
from agent import QLearningAgent


CELL_SIZE = 70
DELAY = 350

WHITE = (245, 245, 245)
BLACK = (40, 40, 40)
BLUE = (60, 120, 230)
GREEN = (40, 190, 100)
RED = (220, 70, 70)
YELLOW = (255, 210, 60)
GRAY = (200, 200, 200)


def main():
    if not os.path.exists("models/q_table.npy"):
        print("Model not found. Run python train.py first.")
        return

    env = MazeEnvironment()
    agent = QLearningAgent()
    agent.load("models/q_table.npy")

    # Test the trained agent without exploration
    state = env.reset()
    path = [state]
    total_reward = 0
    done = False

    for _ in range(200):
        action = agent.choose_action(state, training=False)
        next_state, reward, done = env.step(action)

        total_reward += reward
        state = next_state
        path.append(state)

        if done:
            break

    print("Test route:", path)
    print("Moves:", len(path) - 1)
    print("Total reward:", total_reward)
    print("Reached goal:", done and state == env.goal)

    pygame.init()

    screen = pygame.display.set_mode(
        (env.cols * CELL_SIZE, env.rows * CELL_SIZE)
    )
    pygame.display.set_caption("Reinforcement Learning Maze AI")
    clock = pygame.time.Clock()

    index = 0
    last_move = pygame.time.get_ticks()
    running = True
    font = pygame.font.SysFont("Arial", 22)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    index = 0
                    last_move = pygame.time.get_ticks()

        if index < len(path) - 1:
            now = pygame.time.get_ticks()
            if now - last_move >= DELAY:
                index += 1
                last_move = now

        screen.fill(WHITE)

        for row in range(env.rows):
            for col in range(env.cols):
                cell = (row, col)
                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                color = WHITE

                if cell in env.walls:
                    color = BLACK
                elif cell == env.start:
                    color = RED
                elif cell == env.goal:
                    color = GREEN
                elif cell in path[:index + 1]:
                    color = YELLOW

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

        row, col = path[index]
        center = (
            col * CELL_SIZE + CELL_SIZE // 2,
            row * CELL_SIZE + CELL_SIZE // 2
        )
        pygame.draw.circle(screen, BLUE, center, 18)

        if path[index] == env.goal:
            screen.blit(
                font.render("Goal reached!", True, GREEN),
                (10, 10)
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

