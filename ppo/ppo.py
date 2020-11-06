import numpy as np

class PPO:
    def __init__(self,
                 actor,
                 critic,
                 env,
                 replay_buffer,
                 learning_rate,
                 batch_size,
                 gamma,
                 max_grad_norm,
                 device):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.device = device

    def collect_rollouts(self, n_episodes):
        self.replay_buffer.reset()
        for i in range(n_episodes):
            last_obs = self.env.reset()
            last_done = False
            while not last_done:
                action = self.actor(...)
                log_prob = self.actor(...)
                value = self.critic(...)
                obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(last_obs, action, reward, last_done, value, log_prob)
                last_obs, last_done = obs, done

    def calc_return_and_advantage(self):
        rewards, values = self.replay_buffer.rewards, self.replay_buffer.values
        # Not sure here: What is the return in the last time step?
        expected_return = rewards + self.gamma * np.concatenate([values[1:], values[-1]])
        advantage = expected_return - values
        return expected_return, advantage



