"""
    Name:
    Surname:
    Student ID:
"""

from Environment import Environment
from rl_agents.RLAgent import RLAgent
import numpy as np


class QLearningAgent(RLAgent):
    epsilon: float          # Current epsilon value for epsilon-greedy
    epsilon_decay: float    # Decay ratio for epsilon
    epsilon_min: float      # Minimum epsilon value
    alpha: float            # Alpha value for soft-update
    max_episode: int        # Maximum iteration
    Q: np.ndarray           # Q-Table as Numpy Array

    def __init__(self, env: Environment, seed: int, discount_rate: float, epsilon: float, epsilon_decay: float,
                 epsilon_min: float, alpha: float, max_episode: int):
        """
        Initiate the Agent with hyperparameters.
        :param env: The Environment where the Agent plays.
        :param seed: Seed for random
        :param discount_rate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param epsilon: Initial epsilon value for e-greedy
        :param epsilon_decay: epsilon = epsilon * epsilonDecay after all e-greedy. Less than 1.0
        :param epsilon_min: Minimum epsilon to avoid overestimation. Must be positive or zero
        :param max_episode: Maximum episode for training
        :param alpha: To update Q values softly. 0 < alpha <= 1.0
        """
        super().__init__(env, discount_rate, seed)

        assert epsilon >= 0.0, "epsilon must be >= 0"
        self.epsilon = epsilon

        assert 0.0 <= epsilon_decay <= 1.0, "epsilonDecay must be in range [0.0, 1.0]"
        self.epsilon_decay = epsilon_decay

        assert epsilon_min >= 0.0, "epsilonMin must be >= 0"
        self.epsilon_min = epsilon_min

        assert 0.0 < alpha <= 1.0, "alpha must be in range (0.0, 1.0]"
        self.alpha = alpha

        assert max_episode > 0, "Maximum episode must be > 0"
        self.max_episode = max_episode

        self.Q = np.zeros((self.state_size, self.action_size))

        # If you want to use more parameters, you can initiate below

    def train(self, **kwargs):
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        You will fill the Q-Table with Q-Learning algorithm.

        :param kwargs: Empty
        :return: Nothing
        """

        pass

    def act(self, state: int, is_training: bool) -> int:
        """
        DO NOT CHANGE the name, parameters and return type of the method.

        This method will decide which action will be taken by observing the given state.

        In training, you should apply epsilon-greedy approach. In validation, you should decide based on the Policy.

        :param state: Current State as Integer not Position
        :param is_training: If training use e-greedy, otherwise decide action based on the Policy.
        :return: Action as integer
        """

        return self.rnd.randint(0, self.action_size - 1)
