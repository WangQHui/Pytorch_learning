import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym

#parameters
Batch_size = 32
Lr = 0.001
Epsilon = 0.9
Gamma = 0.9
Target_replace_iter = 100
Memory_capacity = 2000
env = gym.make('Pendulum-v0')  # dueling-dqn test
env = env.unwrapped

N_actions = 25
ACTION_SPACE = 25
N_states = env.observation_space.shape[0]


class DuelingNet(nn.Module):
    def __init__(self):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(N_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, 0.1)
        self.out.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(N_states, 10)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out2 = nn.linear(10, N_actions)
        self.out2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = F.relu(x1)

        x2 = self.fc1(x)
        x2 = F.relu(x2)

        y1 = self.out(x1)
        y2 = self.out2(x2)
        x3 = y1.expand_as(y2) + (y2 - y2.mean(1).expand_as(y2))
        actions_value = x3
        return actions_value

class DuelingDQN(object):
    def __init__(self):
        self.evel_net, self.target_net = DuelingNet(), DuelingNet()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < Epsilon:
            actions_value = self.evel_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
        else:
            action = np.random.randint(0, N_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.FloatTensor(b_memory[:, N_states:N_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states+1:N_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        # q_evel w.r.t the action in experience
        q_eval = self.evel_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn =  DuelingDQN()


print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    acc_r = [0]
    while True:
        a = dqn.choose_action(s)

        f_action = (a - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        # take action
        s_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10  # normalize to a range of (-1, 0)
        acc_r.append(reward + acc_r[-1])  # accumulated reward
        # modify the reward

        dqn.store_transition(s, a, reward, s_)

        if dqn.memory_counter > Memory_capacity:
            dqn.learn()
            if done:
                print('Ep:', i_episode,
                      '| Ep_r:', round(ep_r, 2))

        if done:
            break
        s = s_

env.close()