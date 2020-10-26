import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple
from torch.distributions.normal import Normal


class Policy(nn.Module):
    def __init__(self,
                 parser_args,
                 layer_dims: List = None,
                 non_linearity: nn.Module = nn.Tanh(),
                 eps: float = 1e-6,
                 logger=None):

        if layer_dims is None:
            layer_dims = [parser_args.num_observations, 32, 32]

        super(Policy, self).__init__()

        self.num_intentions = parser_args.num_intentions
        self.num_actions = parser_args.num_actions
        self.eps = eps
        self.lower_log_std_bound = np.log(1e-3)
        self.upper_log_std_bound = np.log(1)

        self.logger = logger

        self.episode_length = parser_args.episode_length

        # Create a model
        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            modules.append(non_linearity)
        modules.append(nn.Linear(layer_dims[-1], 2 * self.num_actions))
        self.model = nn.Sequential(*modules)
        self.model.apply(self.init_weights)  # orthogonal weight initialization
        with torch.no_grad():
            # Scale the last weights to enforce small updates
            list(self.model.parameters())[-2] *= 1e-2

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        pred = self.model(x)
        mean = pred[:, :self.num_actions]
        log_std = pred[:, self.num_actions:]
        assert not torch.isnan(log_std).any()
        log_std = torch.log(1 + torch.exp(log_std) + 1e-6)
        assert not torch.isnan(log_std).any()
        return mean, log_std

    def action_sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates an action sample from the policy network. The output of the network is assumed to be gaussian
        distributed. Let u be a random variable with distribution p(u|s). Since we want our actions to be bound in
        [-1, 1] we apply the tanh function to u, that is a = tanh(u). By change of variable, we have:

        π(a|s) = p(u|s) |det(da/du)| ^-1. Since da/du = diag(1 - tanh^2(u)). We obtain the log likelihood as
        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i)

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            action sample a = tanh(u) and log prob log π(a|s)
        """
        dist = self.normal_dist(mean, log_std)
        normal_action = dist.rsample()  # rsample() employs reparameterization trick
        action = torch.tanh(normal_action)
        normal_log_prob = dist.log_prob(normal_action)
        log_prob = torch.sum(normal_log_prob, dim=-1) - torch.sum(torch.log((1 - action.pow(2) + self.eps)), dim=-1)
        return action, log_prob

    def get_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                     normal_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the log prob of a given action a = tanh(u) and u ~ N(•|μ(s), σ(s)) according to

        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i).

        If u is not given we can reconstruct it with u = tanh^-1(a), since tanh is bijective.

        Args:
            actions: a = tanh(u)
            mean: μ(s)
            log_std: log σ(s)
            normal_actions: u ~ N(•|μ(s), σ(s))

        Returns:
            log π(a|s)
        """
        if normal_actions is None:
            normal_actions = self.inverseTanh(actions)

        normal_log_probs = self.normal_dist(mean, log_std).log_prob(normal_actions)
        log_probs = torch.sum(normal_log_probs, dim=-1) - torch.sum(torch.log(1 - actions.pow(2) + self.eps), dim=-1)
        assert not torch.isnan(log_probs).any()
        return log_probs

    @staticmethod
    def normal_dist(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        """
        Returns a normal distribution.

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            N(u|μ(s), σ(s))
        """
        return Normal(loc=mean, scale=log_std)

    def inverseTanh(self, action: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse of the tanh for the given action
        Args:
            action: a = tanh(u)

        Returns:
            u = tanh^-1(a)
        """
        eps = torch.finfo(action.dtype).eps  # The smallest representable number such that 1.0 + eps != 1.0
        atanh = self.atanh(action.clamp(min=-1. + eps, max=1. - eps))
        assert not torch.isnan(atanh).any()
        return atanh

    @staticmethod
    def atanh(action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (action.log1p() - (-action).log1p())

    def copy_params(self, source_network: nn.Module) -> None:
        """
        Copy the parameters from the source network to the current network.

        Args:
            source_network: Network to copy parameters from

        Returns:
            No return value
        """
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def freeze_net(self) -> None:
        """
        Deactivate gradient
            Computation for the network

        Returns:
            No return value
        """
        for params in self.parameters():
            params.requires_grad = False

    def is_shared(self) -> bool:
        """
        Checks if the network parameter are shared.

        Returns:
            True if shared
        """
        for params in self.parameters():
            if not params.is_shared():
                return False
        return True

    @staticmethod
    def init_weights(module: nn.Module, gain=1.41) -> None:
        """
        Orthogonal initialization of the weights. Sets initial bias to zero.

        Args:
            module: Network to initialize weights.
            gain: Scaling factor

        Returns:
            No return value
        """
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    def has_zero_grads(self):
        for params in self.parameters():
            assert params.grad.sum() == 0

    @property
    def param_norm(self):
        return torch.norm(torch.stack([torch.norm(p.detach()) for p in self.parameters()]))

    @property
    def grad_norm(self):
        return torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.parameters()]))

