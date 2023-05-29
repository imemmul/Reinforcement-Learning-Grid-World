"""
    Name:
    Surname:
    Student ID:
"""

import os.path

import numpy as np

import rl_agents
from Environment import Environment
import time


GRID_DIR = "grid_worlds/"


if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # Type your parameters
    agents = [rl_agents.QLearningAgent(env), rl_agents.SARSAAgent(env)]

    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        print()

        env.reset()

        start_time = time.time_ns()

        agent.train()

        end_time = time.time_ns()

        path, score = agent.validate()

        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        print("*" * 50)
