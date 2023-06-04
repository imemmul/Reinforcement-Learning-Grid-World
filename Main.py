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
from tree_search_agents import AStarAgent
import matplotlib.pyplot as plt

GRID_DIR = "grid_worlds/"


def run_experiment(env, agent, hyperparams, param_name, default_params):
    experiments = []
    actions = ["UP", "LEFT", "DOWN", "RIGHT"]
    for param in hyperparams[param_name]:
        default_params[param_name] = param

        agent_created = agent(env, 42, **default_params)
        
        start_time = time.time_ns()
        td_errors, rewards = agent_created.train(**{"param_name":{param_name}})
        path, score = agent_created.validate()
        end_time = time.time_ns()
        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        experiments.append((param, td_errors, rewards))
        
    plot_experiments(experiments=experiments, agent_name=agent_created.name, exp_name=param_name)
    return experiments, agent_created

def plot_experiments(experiments, agent_name, exp_name):
    fig, axs = plt.subplots(2, figsize=(10, 10))

    for params, td_errors, rewards in experiments:
        axs[0].plot(rewards, label=params)
        axs[1].plot(td_errors, label=params)

    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Mean Reward')
    axs[0].legend(loc='best')
    axs[0].set_title('Hyperparameters Tuning - Reward')

    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Mean TD Error')
    axs[1].legend(loc='best')
    axs[1].set_title('Hyperparameters Tuning - TD Error')

    plt.tight_layout()
    plt.savefig(f"./figures/experiment_{agent_name}_{exp_name}.png")
    plt.figure().clear()

if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # Type your parameters
    # agents = [rl_agents.QLearningAgent(env), rl_agents.SARSAAgent(env)]
    tree_agent = AStarAgent()
    # agents = [rl_agents.QLearningAgent(env, seed=42, discount_rate=1, epsilon=1, epsilon_decay=0, epsilon_min=1, alpha=1, max_episode=100),
    #           rl_agents.SARSAAgent(env, seed=42, discount_rate=1, epsilon=1, epsilon_decay=0, epsilon_min=1, alpha=1, max_episode=100)]
    agents = [rl_agents.QLearningAgent, rl_agents.SARSAAgent]
    # agents = [rl_agents.SARSAAgent(env, seed=42, discount_rate=1, epsilon=0.3, epsilon_decay=0, epsilon_min=1, alpha=1, max_episode=30)]
    # actions_optimal = []
    # total_reward = 0
    # path = []
    # actions_optimal, total_reward, path =  tree_agent.run(env)
    # print(f"Actions: {actions_optimal}")
    # print(f"Score: {total_reward}")
    # print(f"Path: {path}")
    default_params = {
        "discount_rate": 0.9,
        "epsilon": 1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.1,
        "alpha": 0.5,
        "max_episode": 50
    }
    hyperparams = {
        "discount_rate": [1, 0.95, 0.98, 0.99],
        "epsilon": [1, 0.1, 0.01, 0.001],
        "epsilon_decay": [0.995, 0.998, 0.999],
        "epsilon_min": [0.01, 0.001, 0.0001],
        "alpha": [1, 0.1, 0.01, 0.001],
        "max_episode": [10, 20, 30, 40, 50, 100]
    }
    param_names = ["discount_rate", "epsilon", "alpha"]
    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        

        env.reset()

        start_time = time.time_ns()

        # print(agent.run_experiment_convergence(env=env, agent=agent, num_runs=100))
        # avg_td_error, avg_reward = agent.train()
        # print(avg_td_error, avg_reward)
        # print(agent.train())
        for param_name in param_names:
            run_experiment(env=env, agent=agent, param_name=param_name, hyperparams=hyperparams, default_params=default_params)
        
        

        end_time = time.time_ns()


        # print("Actions:", [actions[i] for i in path])
        # print("Score:", score)
        # print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        print("*" * 50)


