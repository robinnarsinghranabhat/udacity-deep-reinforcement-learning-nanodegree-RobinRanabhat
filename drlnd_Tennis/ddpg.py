
import torch
from torch.optim import Adam
import numpy as np
from model import Actor, Critic

import random
from config import Config
from utilities import hard_update



class DDPGAgent:
    """Interacts with and learns from the environment using DDPG method."""

    def __init__(self):
        """Initialize an DDPG Agent object."""
        super(DDPGAgent, self).__init__()
        self.config = Config.getInstance()
        self.actor = Actor(self.config.state_size, self.config.action_size,
                           self.config.seed).to(self.config.device)
        self.critic = Critic(self.config.num_agents * self.config.state_size,
                             self.config.num_agents * self.config.action_size,
                             self.config.seed).to(self.config.device)
        self.target_actor = Actor(
            self.config.state_size, self.config.action_size,
            self.config.seed).to(self.config.device)
        self.target_critic = Critic(self.config.num_agents * self.config.state_size,
                                    self.config.num_agents * self.config.action_size,
                                    self.config.seed).to(self.config.device)
        ## FOR EXPLORATION
        self.noise = OUNoise(self.config.action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.lr_critic,
            weight_decay=self.config.weight_decay)

    def act(self, obs, decay_parameter=0.0):
        """
        Get actions for given state for an agent.
        """
        obs = torch.from_numpy(obs).float().to(self.config.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
        action += decay_parameter * self.noise.sample()
        return action

    def target_act(self, obs, decay_parameter=0.0):
        """
        Get target network actions from an agent
        """
#         obs = obs.to(self.config.device)
        obs = obs.float().contiguous().to(self.config.device)
        ## contiguous for GPU
        
#         import pdb ; pdb.set_trace()
        return self.target_actor(obs) + decay_parameter * self.noise.sample().to(self.config.device)
        
        
        
    def reset(self):
        """Reset the internal state of noise mean(mu)"""
        self.noise.reset()
        
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15,
                 sigma=0.2):
        """Initialize parameters and noise process."""
        self.config = Config.getInstance()
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        random.seed(self.config.seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()

