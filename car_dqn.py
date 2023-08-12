from dqn import DQN
import numpy as np

class CarRacingDQN(DQN):
    #CarRacing specific part of the DQN-agent

    # ** is used for unpacking the model configurations
    def __init__(self, max_negative_rewards=100, **model_config):

        all_actions = np.array([[-1, 0, 0],  [0, 1, 0], [0, 0, 0.5], [0, 0, 0],[1, 0, 0]])

        #Set self parameters
        super().__init__(action_map=all_actions, pic_size=(96, 96), **model_config)

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] > 0 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    def get_random_action(self):
        # give priority to acceleration actions
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)

        return np.random.choice(self.dim_actions, p=action_weights)