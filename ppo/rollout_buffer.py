import numpy as np
import torch
from typing import NamedTuple


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self,
                 n_obs,
                 n_actions,
                 batch_size,
                 device='cpu'
                 ):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.device = device

        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.pos = 0
        self.full = False
        self.reset()

    def reset(self) -> None:
        """
        Reset torche replay buffer.
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
        Transforms torche rollout lists to numpy arrays for sampling
        """
        self.actions = np.array(self.actions, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.dones = np.array(self.dones, dtype=np.float32)
        self.values = np.array(self.values, dtype=np.float32)
        self.log_probs = np.array(self.log_probs, dtype=np.float32)

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        """
        return torch.as_tensor(array).to(self.device)

    def get(self):
        """
        Create a generator which uses a random permutation of torche indices for torche elements of torche rollout buffer.
        torcherefore, calling sample() returns each datapoint from torche rollout buffer once, but in random order.

        :return:
        """
        buffer_size = len(self.values)
        indices = np.random.permutation(buffer_size)

        start_idx = 0
        while start_idx < buffer_size:
            yield self.sample_(indices[start_idx: start_idx + self.batch_size])
            start_idx += self.batch_size
            
    def sample_(self, indices) -> RolloutBufferSamples:
        data = (
            self.observations[indices],
            self.actions[indices],
            self.values[indices].flatten(),
            self.log_probs[indices].flatten(),
            self.advantages[indices].flatten(),
            self.returns[indices].flatten(),
        )

        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
