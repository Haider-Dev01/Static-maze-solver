import gym
from maze_environment import MazeEnv
from agent import Agent
import pygame

def train_agent(env, agent, episodes=100):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:

            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            env.render()  # Display the environment after each step
            pygame.time.delay(100)  # Slow down the simulation by 300 ms
        print(f"Episode {episode + 1}: Total reward = {total_reward}")

def test_agent(env, agent):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render()  # Display the environment after each step
        pygame.time.delay(100)  # Slow down the simulation by 300 ms
        print(f"Action: {action}, Reward: {reward}, State: {next_state}")
        if done:
            print("Goal reached!")

if __name__ == "__main__":
    # Initialize the custom maze environment (Gym)
    env = MazeEnv(grid_size=(10, 10), goal=(9, 9), obstacles=[(2, 2), (3, 1), (1, 3),(5, 4),(10,7)])

    # Initialize the agent
    agent = Agent(actions=[0, 1, 2, 3], alpha=0.1, gamma=0.9, epsilon=0.9)

    # Train the agent
    print("Training the agent...")
    train_agent(env, agent, episodes=100)

    # Test the trained agent
    print("\nTesting the trained agent...")
    test_agent(env, agent)

    # Close the environment properly after the simulation
    env.close()  # This ensures pygame.quit() is called and window closes gracefully
