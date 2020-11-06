import numpy as np


class RolloutBuffer:
    def __init__(self,
                 actor,
                 critic,
                 buffer_size,
                 n_obs,
                 n_actions,
                 ):

        self.actor = actor
        self.critic = critic
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.observations = np.zeros(self.buffer_size + self.n_obs, dtype=np.float32)

        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.pos = 0
        self.full = False
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.actions = ()
        self.rewards = ()
        self.returns = ()
        self.dones = ()
        self.values = ()
        self.log_probs = ()
        self.advantages = ()

    def push(self, obs, action, reward, done, value, log_prob):
        self.observations.append(np.array(obs).copy()).append(np.array(action).copy())
        self.rewards.append(np.array(reward).copy())
        self.dones.append(np.array(done).copy())
        self.values.append(value.clone().cpu().numpy().flatten())
        self.log_probs.append(log_prob.clone().cpu().numpy())




