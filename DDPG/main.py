import argparse
from tensorboardX import SummaryWriter
import os
import gym
import numpy as np
import itertools
import random
import torch
from ddpg import DDPG
from ounoise import OUNoise
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--test_interval', type=int, default=5000, metavar='N',
                    help='Test Steps')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
args = parser.parse_args()

env = gym.make(args.env_name)

i=0
while True:
    if os.path.exists('run/{}/ddpg/{}'.format(args.env_name, i)):
        i += 1
    else:
        break
writer = SummaryWriter('run/{}/ddpg/{}'.format(args.env_name, i))
seed=random.randint(0,1000000)
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

agent = DDPG(args.gamma, args.tau, args.hidden_size,
                  env.observation_space.shape[0], env.action_space)

memory = ReplayMemory(args.replay_size, seed)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None


total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    state = torch.Tensor([env.reset()])

    if args.ou_noise: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    done=False
    while not done:
        if total_numsteps > args.num_steps:
            break

        action = agent.select_action(state, ounoise)
        next_state, reward, done, _ = env.step(action)
        total_numsteps += 1

        action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                agent.update_parameters(memory, args.batch_size, updates)

        if total_numsteps % args.test_interval == 0:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward= 0
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= 10
            writer.add_scalar('score/score', avg_reward, total_numsteps)

            print("----------------------------------------")
            print("Total Numsteps: {}, Avg. Reward: {}".format(total_numsteps, round(avg_reward, 2)))
            print("----------------------------------------")
            break

env.close()
