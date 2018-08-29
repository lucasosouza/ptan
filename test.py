# test.py

import ptan
import gym
from torch import nn, optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4 # specify how many episodes will be used for training

class PGN(nn.Module):
  """ Model exactly similar to cross entropu"""
  
  def __init__(self, input_size, n_actions):
    super(PGN, self).__init__()
    
    self.net = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      nn.Linear(128, n_actions)
    )
    # output logits, not apply softmax
    # use PyTorch log_softmax at output, more numerically stable
    
  def forward(self, x):
    return self.net(x)
  
  def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
      # discount existing by gamma
      sum_r *= GAMMA
      # sum new reward
      sum_r += r
      # append to list of rewards
      res.append(sum_r)
    # put list in right order before returning
    return list(reversed(res))

env = gym.make("CartPole-v0")
net = PGN(env.observation_space.shape[0], env.action_space.n)
agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                              apply_softmax=True) 
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
