"""Compatibility import for the original project layout."""

from static_maze_solver.agent import QLearningAgent

Agent = QLearningAgent

__all__ = ["Agent", "QLearningAgent"]
