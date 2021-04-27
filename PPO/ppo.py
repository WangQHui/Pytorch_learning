import gym
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        # 初始化ppo模型，包括超参数
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        # 初始化超参数
        self.__init_hyperparameters(hyperparameters)

        # 环境信息
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # 初始化actor和critic网络
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # 初始化actor和critic的优化器
        self.act_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # 初始化用于查询actor的动作的协方差矩阵
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)  # 返回大小为sizes,单位值为fill_value的矩阵
        self.cov_mat = torch.diag(self.cov_var)

        # 此记录器将帮助我们打印出每次迭代的摘要
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
        }





