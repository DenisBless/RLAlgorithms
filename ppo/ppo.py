import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 env,
                 rollout_buffer,
                 learning_rate,
                 n_epochs,
                 batch_size,
                 gamma,
                 max_grad_norm,
                 device,
                 optimizer: torch.optim = torch.optim.Adam,
                 eps=0.2):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.eps = eps
        self.rollout_buffer = rollout_buffer # todo maybe instancizte here
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.batch_size = batch_size
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor_optimizer = optimizer(self.actor.paramters())
        self.critic_optimizer = optimizer(self.critic.paramters())

    def collect_rollouts(self, n_episodes):
        self.rollout_buffer.reset()
        for i in range(n_episodes):
            last_obs = self.env.reset()
            last_done = False
            while not last_done:
                action = self.actor(...)
                log_prob = self.actor(...)
                value = self.critic(...)
                obs, reward, done, _ = self.env.step(action)
                self.rollout_buffer.add_transition_vars(last_obs, action, reward, last_done, value, log_prob)
                last_obs, last_done = obs, done
            returns, advantages = self.calc_return_and_advantage()
            self.rollout_buffer.add_episode_vars(returns, advantages)

    def calc_return_and_advantage(self):
        #Todo No GAE implemented yet
        rewards, values = self.rollout_buffer.rewards, self.rollout_buffer.values
        # Not sure here: What is the return in the last time step?
        expected_return = rewards + self.gamma * np.concatenate([values[1:], values[-1]])
        advantage = expected_return - values
        return expected_return, advantage

    def train(self):
        self.rollout_buffer.to_numpy()

        for epoch in range(self.n_epochs):
            for batch_data in self.rollout_buffer.get():
                # Compute losses
                actions = batch_data.actions
                log_prob, entropy = self.actor.calc_logprob_and_ent(actions)  # TODO actor has to implement this fn
                ratio = torch.exp(log_prob - batch_data.old_log_prob)

                advantages = batch_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantage

                policy_loss_p1 = ratio * advantages
                policy_loss_p2 = torch.clamp(ratio, min=1 - self.eps, max=1 + self.eps) * advantages
                policy_loss = -torch.min(policy_loss_p1, policy_loss_p2).mean()

                values = self.critic(obs=batch_data.observations, actions=batch_data.actions)
                value_loss = F.mse_loss(batch_data.returns, values)

                # Optimization steps
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.critic.parameter(), max_norm=self.max_grad_norm)
                self.critic_optimizer.step()
