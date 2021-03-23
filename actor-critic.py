import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

gamma = 0.99
lr = 0.02
betas = (0.9, 0.999)
random_seed = 543
torch.manual_seed(random_seed)
env = gym.make('LunarLander-v2')
env.seed(random_seed)

class ActorCritic(nn.module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, gamma=0.99):
        # calculating discounted rewards
        rewards = []
        dis_reward = 0
        for rewards in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.means()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)

        return loss

    def clearMenory(self):
        del self.logprobs[:]
        del self.state_value[:]
        del self.rewards[:]


print('\nCollecting experience...')
for i_episode in range(0, 10000):
    state = env.reset()
    for t in range(10000):
        action = dqn.choose_action(state)
        state, reward, done, _ = env.step(action)
        policy.rewards.append(reward)
        running_reward += reward
        if render and i_episode > 1000:
            env.render()
        if done:
            break

    # Updating the policy :
    optimizer.zero_grad()
    loss = policy.calculateLoss(gamma)
    loss.backward()
    optimizer.step()
    policy.clearMemory()

    # saving the model if episodes > 999 OR avg reward > 200
    # if i_episode > 999:
    #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))

    if running_reward > 4000:
        torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        print("########## Solved! ##########")
        test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        break

    if i_episode % 20 == 0:
        running_reward = running_reward / 20
        print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
        running_reward = 0

