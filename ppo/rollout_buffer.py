import numpy as np


class RolloutBuffer:
    def __init__(self,
                 n_obs,
                 n_actions,
                 ):
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.pos = 0
        self.full = False
        self.reset()

    def reset(self) -> None:
        """
        Reset the replay buffer.
        """
        self.pos = 0
        self.actions = ()
        self.rewards = ()
        self.dones = ()
        self.values = ()
        self.log_probs = ()
        self.returns = None
        self.advantages = None

    def add_transition_vars(self, obs, action, reward, done, value, log_prob) -> None:
        self.observations.append(np.array(obs).copy()).append(np.array(action).copy())
        self.rewards.append(np.array(reward).copy())
        self.dones.append(np.array(done).copy())
        self.values.append(value.clone().cpu().numpy().flatten())
        self.log_probs.append(log_prob.clone().cpu().numpy())

    def add_episode_vars(self, returns, advantages) -> None:
        self.returns = returns
        self.advantages = advantages

    def to_numpy(self) -> None:
        """
        Transforms the rollout lists to numpy arrays for sampling
        """
        self.actions = np.array(self.actions, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.dones = np.array(self.dones, dtype=np.float32)
        self.values = np.array(self.values, dtype=np.float32)
        self.log_probs = np.array(self.log_probs, dtype=np.float32)

    def sample(self):
        """
        Create a generator which uses a random permutation of the indices for the elements of the rollout buffer.
        Therefore, calling sample() returns each datapoint from the rollout buffer once, but in random order.

        :return:
        """
        ...
