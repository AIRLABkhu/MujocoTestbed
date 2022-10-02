import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.utils import *
from common.networks import *


class Agent(object):
    """An implementation of the Twin Delayed DDPG (TD3) agent."""

    def __init__(self,
                 env,
                 args,
                 device,
                 obs_dim,
                 act_dim,
                 act_limit,
                 steps=0,
                 expl_before=2000,
                 train_after=1000,
                 gamma=0.99,
                 act_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 hidden_sizes=(128, 128),
                 buffer_size=int(1e4),
                 batch_size=64,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 eval_mode=False,
                 policy_losses=list(),
                 qf_losses=list(),
                 logger=dict(),
                 ):

        self.env = env
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.steps = steps
        self.expl_before = expl_before
        self.train_after = train_after
        self.gamma = gamma
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.hidden_sizes = hidden_sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.eval_mode = eval_mode
        self.policy_losses = policy_losses
        self.qf_losses = qf_losses
        self.logger = logger

        # Main network
        self.policy = MLP(self.obs_dim, self.act_dim, self.act_limit,
                          hidden_sizes=self.hidden_sizes,
                          output_activation=torch.tanh,
                          use_actor=True).to(self.device)
        self.qf1 = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        self.qf2 = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        # Target network
        self.policy_target = MLP(self.obs_dim, self.act_dim, self.act_limit,
                                 hidden_sizes=self.hidden_sizes,
                                 output_activation=torch.tanh,
                                 use_actor=True).to(self.device)
        self.qf1_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)
        self.qf2_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(self.device)

        # Initialize target parameters to match main parameters
        hard_target_update(self.policy, self.policy_target)
        hard_target_update(self.qf1, self.qf1_target)
        hard_target_update(self.qf2, self.qf2_target)

        # Concat the Q-network parameters to use one optim
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=self.qf_lr)



    def select_action(self, obs,evaluate=False):

        if evaluate==False:
            action = self.policy(obs).detach().cpu().numpy()
            action += self.act_noise * np.random.randn(self.act_dim)
            return np.clip(action, -self.act_limit, self.act_limit)
        else:
            action = self.policy(obs)
            action = action.detach().cpu().numpy()
            return action

    def update_parameters(self, replay_buffer):

        batch = replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        if 0:  # Check shape of experiences
            print("obs1", obs1.shape)
            print("obs2", obs2.shape)
            print("acts", acts.shape)
            print("rews", rews.shape)
            print("done", done.shape)

        # Prediction Q1(s,𝜇(s)), Q1(s,a), Q2(s,a)
        q1_pi = self.qf1(obs1, self.policy(obs1))
        q1 = self.qf1(obs1, acts).squeeze(1)
        q2 = self.qf2(obs1, acts).squeeze(1)

        # Target policy smoothing, by adding clipped noise to target actions
        pi_target = self.policy_target(obs2)
        epsilon = torch.normal(mean=0, std=self.target_noise, size=pi_target.size()).to(self.device)
        epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(self.device)
        pi_target = torch.clamp(pi_target + epsilon, -self.act_limit, self.act_limit).to(self.device)

        # Min Double-Q: min(Q1‾(s',𝜇(s')), Q2‾(s',𝜇(s')))
        min_q_pi_target = torch.min(self.qf1_target(obs2, pi_target),
                                    self.qf2_target(obs2, pi_target)).squeeze(1).to(self.device)

        # Target for Q regression
        q_backup = rews + self.gamma * (1 - done) * min_q_pi_target
        q_backup.to(self.device)

        if 0:  # Check shape of prediction and target
            print("pi_target", pi_target.shape)
            print("epsilon", epsilon.shape)
            print("q1", q1.shape)
            print("q2", q2.shape)
            print("min_q_pi_target", min_q_pi_target.shape)
            print("q_backup", q_backup.shape)

        # TD3 losses
        policy_loss = -q1_pi.mean()
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())
        qf_loss = qf1_loss + qf2_loss

        # Delayed policy update
        if self.steps % self.policy_delay == 0:
            # Update policy network parameter
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Polyak averaging for target parameter
            soft_target_update(self.policy, self.policy_target)
            soft_target_update(self.qf1, self.qf1_target)
            soft_target_update(self.qf2, self.qf2_target)

        # Update two Q-network parameter
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.qf_losses.append(qf_loss.item())
