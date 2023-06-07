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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import PIL
import imageio

GRID_DIR = "grid_worlds/"


def run_experiment(env, agent, hyperparams, param_name, default_params):
    experiments = []
    actions = ["UP", "LEFT", "DOWN", "RIGHT"]
    for param in hyperparams[param_name]:
        default_params[param_name] = param
        agent_created = agent(env, 42, **default_params)
        print(f"Running {agent_created.name} with {param_name}={param}")
        print(f"All hyperparameters: {default_params}")
        start_time = time.time_ns()
        kwargs = {"param_name":param_name, "param_value": param}
        td_errors, rewards, q_values = agent_created.train(**kwargs)
        path, score = agent_created.validate()
        end_time = time.time_ns()
        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)
        plot_training(q_values, agent_created, param_name=param_name, param_value=param)
        experiments.append((param, td_errors, rewards))
        
    plot_experiments(experiments=experiments, agent_name=agent_created.name, exp_name=param_name)
    return experiments, agent_created

def plot_training(q_values, agent, param_name, param_value):
    filenames = []
    colors = ["red", "green"]
    cmap = LinearSegmentedColormap.from_list("custom", colors)
    # TODO show goals and starting position on grid
    # goals = [tuple(agent.env.to_position(goal)) for goal in agent.env.get_goals()]
    # starting_pos = agent.env.starting_position
    # for i, q in enumerate(q_values):
    max_Q_values = np.max(q_values, axis=1).reshape((agent.env.grid_size, agent.env.grid_size))
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(max_Q_values, annot=True, fmt=".2f", cmap=cmap, square=True, ax=ax)
    ax. set_title(f'Maximum Q-values - Episode')
    # filenames.append(f"./temp_figures/heatmap_{i}.png")
    plt.savefig(f'./figures/experiment_{agent.name}_{param_name}_{param_value}.png')  # Save each figure with a different name
        # plt.close(fig)

    # Below code gives GIF.
    # images = [imageio.imread(filename) for filename in filenames]
    # imageio.mimsave(f'./figures/training_heatmap_{agent.name}_{param_name}_{param_value}.gif', images, duration=0.1)

def plot_experiments(experiments, agent_name, exp_name):
    fig, axs = plt.subplots(2, figsize=(10, 10))

    for params, td_errors, rewards in experiments:
        axs[0].plot(rewards, label=params)
        axs[1].plot(td_errors, label=params)

    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Rewards')
    axs[0].legend(loc='best')
    axs[0].set_title('Hyperparameters Tuning - Reward')

    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('TD Error')
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
  
    default_params = { # TODO tuning the hyperparameters offers optimal
        "discount_rate": 1,
        "epsilon": 1,
        "epsilon_decay": 0.001,
        "epsilon_min": 0.01,
        "alpha": 0.5,
        "max_episode": 100
    }
    hyperparams = {
        "discount_rate": [1, 0.95, 0.98, 0.99],
        "epsilon": [1, 0.1, 0.01, 0.001],
        "epsilon_decay": [0.001, 0.00001, 0.00001],
        "epsilon_min": [0.01, 0.001],
        "alpha": [1, 0.1, 0.01, 0.001],
        "max_episode": [10, 20, 30, 40, 50, 100, 200]
    }
    param_names = ["discount_rate", "epsilon", "epsilon_decay", "epsilon_min", "alpha"]
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
            run_experiment(env=env, agent=agent, param_name=param_name, hyperparams=hyperparams, default_params=default_params.copy())
        
        

        end_time = time.time_ns()


        # print("Actions:", [actions[i] for i in path])
        # print("Score:", score)
        # print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        print("*" * 50)


