import torch
from torch.optim import Adam
import torch.nn.functional as F
from model import DeterministicPolicy,QNetwork_DDPG

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space
        self.device = torch.device("cuda")

        self.actor = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.actor_target = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = QNetwork_DDPG(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_target = QNetwork_DDPG(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mu = self.actor(state)

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(self.device)

        return mu.detach().cpu().numpy()[0]


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device).squeeze()
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device).squeeze()
        action_batch = torch.FloatTensor(action_batch).to(self.device).squeeze()
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)


        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)
        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))