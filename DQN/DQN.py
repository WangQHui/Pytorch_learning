import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym

# parameters
Batch_size = 32
Lr = 0.001
Epsilon = 0.9
Gamma = 0.9
Target_replace_iter = 100
Memory_capacity = 2000
env = gym.make('CartPole-v0')  # dqn test
env = env.unwrapped
N_actions = env.action_space.n
N_states = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(10, N_actions)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # 建立 target net 和 eval net 还有 memory
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((Memory_capacity, N_states*2 + 2))  # 初始化记忆库
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, x):
        # 根据环境观测值选择动作的机制
        x = Variable(torch.unsqueeze((torch.FloatTensor(x), 0)))
        # 这里只输入一个 sample
        if np.random.uniform() < Epsilon:  # 选最优动作
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # 选随机动作
            action = np.ramdom.randint(0, N_actions)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.FloatTensor(b_memory[:, N_states:N_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states+1:N_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_evel = self.evel_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_evel, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    while True:  # 显示实验动画
        env.render()
        ep_r = 0
        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, done, info = env.step(a)

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dat = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > Memory_capacity:
            dqn.learn()
            if done:
                print('Ep:', i_episode,
                     '| Ep_r:', round(ep_r, 2))

        if done:
            break
        s = s_
env.close()




