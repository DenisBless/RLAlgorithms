import numpy as np


class RolloutBuffer:
    def __init__(self,
                 buffer_size,
                 n_obs,
                 n_actions,
                 ):
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.observations = np.zeros(self.buffer_size + self.n_obs, dtype=np.float32)

        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.reset()

    def reset(self) -> None:
        self.actions = np.zeros((self.buffer_size, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)

    def push(self):
        ...  # todo
