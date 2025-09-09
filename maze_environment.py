import gym
from gym import spaces
import numpy as np
import pygame

class MazeEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), goal=(4, 4), obstacles=None):
        super(MazeEnv, self).__init__()

        self.grid_size = grid_size  # Maze size (rows, columns)
        self.goal = goal  # Goal position
        self.agent_position = (0, 0)  # Start position
        self.obstacles = obstacles if obstacles else [(2, 2), (3, 1), (1, 3)]  # Default obstacles
        
        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Discrete(grid_size[0] * grid_size[1])  # Flattened grid size

        # Pygame setup
        pygame.init()
        self.screen_size = (600, 600)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.cell_size = self.screen_size[0] // self.grid_size[0]  # Calculate cell size
        self.colors = {
            "empty": (255, 255, 255),  # White (path)
            "obstacle": (50, 50, 50),  # Dark Gray (walls)
            "goal": (0, 255, 0),       # Green (goal)
            "agent": (255, 0, 0),       # Red (agent)
            "grid_line": (0, 0, 0)     # Black (grid lines)
        }

    def reset(self):
        """Reset the maze to the initial state."""
        self.agent_position = (0, 0)  # Reset to starting position
        return self.agent_position

    def step(self, action):
        """Take an action and update the state."""
        x, y = self.agent_position
        if action == 0:  # UP
            if x > 0 and (x - 1, y) not in self.obstacles:
                self.agent_position = (x - 1, y)
        elif action == 1:  # DOWN
            if x < self.grid_size[0] - 1 and (x + 1, y) not in self.obstacles:
                self.agent_position = (x + 1, y)
        elif action == 2:  # LEFT
            if y > 0 and (x, y - 1) not in self.obstacles:
                self.agent_position = (x, y - 1)
        elif action == 3:  # RIGHT
            if y < self.grid_size[1] - 1 and (x, y + 1) not in self.obstacles:
                self.agent_position = (x, y + 1)

        # Check if the agent reaches the goal
        reward = -1
        done = False
        if self.agent_position == self.goal:
            reward = 100  # Reward for reaching the goal
            done = True

        return self.agent_position, reward, done, {}

    def render(self, mode='human'):
        """Render the maze environment using Pygame."""
        self.screen.fill(self.colors["empty"])  # Fill the screen with the empty color

        # Draw the obstacles
        for (x, y) in self.obstacles:
            pygame.draw.rect(self.screen, self.colors["obstacle"],
                             (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))

        # Draw the goal
        goal_x, goal_y = self.goal
        pygame.draw.rect(self.screen, self.colors["goal"],
                         (goal_y * self.cell_size, goal_x * self.cell_size, self.cell_size, self.cell_size))

        # Draw the agent
        agent_x, agent_y = self.agent_position
        pygame.draw.rect(self.screen, self.colors["agent"],
                         (agent_y * self.cell_size, agent_x * self.cell_size, self.cell_size, self.cell_size))

        # Draw grid lines
        for row in range(self.grid_size[0] + 1):
            pygame.draw.line(self.screen, self.colors["grid_line"], (0, row * self.cell_size), (self.screen_size[0], row * self.cell_size), 2)
        for col in range(self.grid_size[1] + 1):
            pygame.draw.line(self.screen, self.colors["grid_line"], (col * self.cell_size, 0), (col * self.cell_size, self.screen_size[1]), 2)

        pygame.display.flip()  # Update the display

    def close(self):
        """Close the Pygame window."""
        pygame.quit()  # Make sure to call pygame.quit() to properly close the window.
