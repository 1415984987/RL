import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
from parl.utils import logger
from parl.algorithms import DQN
import random
import collections

# from ple.games.pixelcopter import Pixelcopter
from ple.games.pong import Pong
from ple import PLE

import os 
os.environ["SDL_VIDEODRIVER"] = "dummy"

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 2000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 1024  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

LEARNING_RATE = 0.00002 # 学习率

class Model(parl.Model):
    def __init__(self, act_dim):

        # hid1_size = act_dim*20
        hid1_size = 200
        # hid2_size = act_dim*20
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid1_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)
    def value(self, obs):

        h = self.fc1(obs)
        h = self.fc2(h)
        Q = self.fc3(h)
        return Q

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 100  # 每隔100个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔100个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

def run_episode(ple_env, agent, rpm):
    total_reward = 0
    ple_env.reset_game()
    obs = list(ple_env.getGameState().values())
    step = 0
    while True:
        step += 1
        action_index = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        action = ple_env.getActionSet()[action_index]
        # 行动
        reward = ple_env.act(action)
        next_obs = list(ple_env.getGameState().values())
        done = ple_env.game_over()
        rpm.append((obs, action_index, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(ple_env, agent, render=False):
    eval_reward = []
    for i in range(5):
        ple_env.reset_game()
        obs = list(ple_env.getGameState().values())
        episode_reward = 0
        while True:
            action_index = agent.predict(obs)  # 选取最优动作
            action = ple_env.getActionSet()[action_index]
            reward = ple_env.act(action)
            obs = list(ple_env.getGameState().values())
            episode_reward += reward
            # if render:
            #     ple_env.getScreenRGB()
            if ple_env.game_over():
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 创建环境
game = Pong(cpu_speed_ratio=0.3)
# game = Pong()
pong = PLE(game, display_screen=True, force_fps=True)
# 根据parl框架构建agent
print(pong.getActionSet())
action_dim = len(pong.getActionSet())
obs_shape = len(pong.getGameState())
print(pong.getGameState())
# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池


model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape,
    act_dim=action_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
save_path = './model_pong_12.6.ckpt'
agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(pong, agent, rpm)

max_episode = 10000
test_reward = []
score = -10
best_episode = 0

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    # print(episode)
    total_reward = run_episode(pong, agent, rpm)
    if episode % 10 == 0:
        logger.info('episode:{}    e_greed:{}   train_reward:{}'.format(episode, agent.e_greed, total_reward))
    # test part
    if episode % 100 == 0:
        eval_reward = evaluate(pong, agent, render=False)  # render=True 查看显示效果
        logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(episode, agent.e_greed, eval_reward))
        test_reward.append(eval_reward)
        if eval_reward >= score:
            score = eval_reward
            best_episode = episode
            agent.save('./model_pong_best.ckpt')
            print('save model_best')
        print('model_best：',best_episode,score)
    episode += 1
# 训练结束，保存模型
save_path = './pong_model.ckpt'
agent.save(save_path)
