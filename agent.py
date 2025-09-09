import numpy as np
import random

class Agent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table

    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration: random action
        else:
            # Exploitation: choose action with the highest Q-value
            if state not in self.q_table:
                self.q_table[state] = {action: 0.0 for action in self.actions}
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}
        
        old_q_value = self.q_table[state][action]
        future_q_value = max(self.q_table[next_state].values())
        
        # Update Q-value
        self.q_table[state][action] = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)

    def visualize_q_table(self):
        """Visualize the Q-table (optional)."""
        for state, actions in self.q_table.items():
            print(f"State {state}: {actions}")
