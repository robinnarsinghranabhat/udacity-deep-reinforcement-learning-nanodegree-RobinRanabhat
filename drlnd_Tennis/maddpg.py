

import torch
import random
import numpy as np
import torch.nn.functional as F
from ddpg import DDPGAgent

from collections import deque, namedtuple
from config import Config
from utilities import soft_update


class MADDPGAgent:
    """Interacts and learns from the environment using multiple DDPG agents"""

    def __init__(self):
        """Initialize a MADDPG Agent object."""
        super(MADDPGAgent, self).__init__()
        
        
        self.config = Config.getInstance()
        self.memory = ReplayBuffer()
        self.action_num = self.config.action_size * self.config.num_agents
        self.t_step = 0

        self.maddpg_agent = [DDPGAgent()
                             for _ in range(self.config.num_agents)]

        
        

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(
            self.maddpg_agent, obs_all_agents)]
        return np.concatenate(actions)

    

    def update_act(self, obs_all_agents, agent_num, decay_parameter=0.0):
        """
        get target network actions from all the agents in the MADDPG object
        """
        actions = []
        for an, ddpg_agent in enumerate(self.maddpg_agent):
            obs = obs_all_agents[:, an, :].to(self.config.device)
            
#             import pdb; pdb.set_trace()
            acn = ddpg_agent.actor(
                obs.contiguous()) + decay_parameter * ddpg_agent.noise.sample().to(self.config.device)
            if an != agent_num:
                acn = acn.detach()
            actions.append(acn)
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """
        get target network actions from all the agents in the MADDPG object
        """
        target_actions = [ddpg_agent.target_act(
            obs_all_agents[:, an, :], noise) for an, ddpg_agent in enumerate(self.maddpg_agent)]
        return target_actions

    def step(self, _states, _actions, _rewards, _next_states, _dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        states_full = np.reshape(_states, newshape=(-1))
        next_states_full = np.reshape(_next_states, newshape=(-1))
        self.memory.add(_states, states_full, _actions, _rewards, _next_states,
                        next_states_full,  _dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every

        if self.t_step == 0:
            if len(self.memory) > self.config.batch_size:
                for a_i in range(self.config.num_agents):
                    samples = self.memory.sample()
                    for __ in range(3):
                        self.update(samples, a_i)
                    self.update_targets()


    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        # ---------------------------- update critic ---------------------- #
        # self.update_critic(samples, agent_number)
        states, states_full, actions, rewards, next_states, next_states_full, dones = samples
        agent = self.maddpg_agent[agent_number]


        agent.critic_optimizer.zero_grad()
        # ---------------------------- update critic ---------------------- #
        actions_next = self.target_act(next_states)
        actions_next = torch.cat(actions_next, dim=1)

        Q_target_next = agent.target_critic(next_states_full, actions_next)
        Q_targets = rewards[:, agent_number].view(-1, 1) + self.config.gamma * \
            Q_target_next * (1 - dones[:, agent_number].view(-1, 1))
        Q_expected = agent.critic(
            states_full, actions.reshape(-1, self.action_num))
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------- #
        agent.actor_optimizer.zero_grad()
        actions_pred = self.update_act(states, agent_number)
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -agent.critic(states_full, actions_pred).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()







    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor,
                        ddpg_agent.actor, self.config.tau)
            soft_update(ddpg_agent.target_critic,
                        ddpg_agent.critic, self.config.tau)

    def reset(self):
        """Resets weight of all agents"""
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
            
            
            
## same as in DQN Buffer

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self):
        """Initialize a ReplayBuffer object."""
        self.config = Config.getInstance()
        self.memory = deque(maxlen=int(self.config.buffer_size))
        self.batch_size = int(self.config.batch_size)
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "state_full", "action",
                                     "reward", "next_state",
                                     "next_state_full",
                                     "done"])
        random.seed(self.config.seed)

    def add(self, state, state_full, action, reward, next_state,
            next_state_full, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_full, action, reward, next_state,
                            next_state_full, done)
        self.memory.append(e)

    def convert_to_tensor(self, attributes_list):
        """Convert a list to tensor"""
        return torch.from_numpy(np.array(attributes_list)).float().to(self.config.device)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.convert_to_tensor(
            [e.state for e in experiences if e is not None])
        states_full = self.convert_to_tensor(
            [e.state_full for e in experiences if e is not None])
        actions = self.convert_to_tensor(
            [e.action for e in experiences if e is not None])
        rewards = self.convert_to_tensor(
            [e.reward for e in experiences if e is not None])
        next_states = self.convert_to_tensor(
            [e.next_state for e in experiences if e is not None])
        next_states_full = self.convert_to_tensor(
            [e.next_state_full for e in experiences if e is not None])
        dones = self.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.uint8))

        return (states, states_full, actions, rewards, next_states,
                next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

