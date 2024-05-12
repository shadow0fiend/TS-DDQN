"""
Network:与网络搭建相关的类
Parameters:与参数有关的类
ReplayBuffer:与经验回放池有关的类
Main:主逻辑类
Situation:与强化学习有关的类
"""

import numpy as np
import torch

from Data import DataFactory
from ReplayBuffer import ReplayBuffer
from Parameters import Parameters
from Network import p1_Net,p2_Net
from Situation import Situation

class Run():
    def __init__(self):
        self.init()     #初始化各种类的实例化对象

    def init(self):
        self.parameters = Parameters()
        self.p1_reply_buffer = ReplayBuffer(self.parameters.capacity_history)
        self.p2_reply_buffer = ReplayBuffer(self.parameters.capacity_history)
        self.data_factory = DataFactory()
        self.data_factory.generate("../data/t0.txt",self.parameters.jobs,self.parameters.process,self.parameters.machines)
        self.situation = Situation(self.parameters)

        self.p1_online = p1_Net()  # online获取下标
        self.p1_target = p1_Net()  # target获取Q值
        self.p2_online = p2_Net()
        self.p2_target = p2_Net()

        self.optimizer_1, self.loss_function_1 = self.parameters.set_loss(self.p1_target.parameters(), self.parameters.learning_rate)
        self.optimizer_2, self.loss_function_2 = self.parameters.set_loss(self.p2_target.parameters(),self.parameters.learning_rate)

    def start(self):
        for epoch in range(self.parameters.epochs):
            self.p1_target.load_state_dict(torch.load('./net/p1_target.pth'))
            self.p2_target.load_state_dict(torch.load('./net/p2_target.pth'))

            for p in range(self.parameters.jobs * self.parameters.process): # 总共要遍历的工序
                if p == 0:  # 第一次就随机选择
                    self.situation.initialProcess()
                    continue
                # 计算的当前的state
                self.situation.getState()
                # 将state输入到第一阶段的双层网络中，获取奖励函数
                p1_online_ret = torch.argmax(self.p1_online.forward(self.situation.stateVector)).item()
                p1_target_ret = self.p1_target.forward(self.situation.stateVector)[p1_online_ret].item()

                # 将state输入到第二阶段的双层网络中，获取行为
                p2_online_ret = torch.argmax(self.p2_online.forward(self.situation.stateVector)).item()
                p2_target_ret = self.p2_target.forward(self.situation.stateVector)[p2_online_ret].item()

                # 执行行为，（更新完工工序，未完工工序，工序对应机器序列）
                jobNum, machineNum = self.situation.Action(p2_online_ret,self.parameters.epsilon)

                print(f"第{epoch + 1}轮 第{p+1}道工序 总共{self.parameters.jobs * self.parameters.process}道工序: 将工件{jobNum+1}分配给机器{machineNum+1}加工")

                # 保存久的状态值，重新计算state
                old_state = self.situation.stateVector
                self.situation.caculate()
                self.situation.getState()
                new_state = self.situation.stateVector

                # 计算奖励
                reward = self.situation.reward(p1_online_ret,old_state,new_state)

                # 将（old_state，g，reward，new_state）存入经验回放池中
                self.p1_reply_buffer.push(old_state,p1_online_ret,reward,new_state)
                self.p2_reply_buffer.push(old_state,p2_online_ret,reward,new_state)

                # 经验回放池容量达到一定数量以后，随机从经验回放池中抽取数据，准备做梯度下降，数量不够就不做了
                if len(self.p1_reply_buffer.buffer) >= self.parameters.minibatch_size:
                    for i in range(self.parameters.minibatch_size): # 每次取出一个
                        state, action, reward, next_state, done = self.p1_reply_buffer.choice_one()
                        # 使用这一个来进行梯度下降
                        y_ = reward + self.parameters.gamma * self.p1_target(next_state)[action].item()
                        y = torch.max(self.p1_target(state)).item()
                        loss = self.loss_function_1(torch.tensor(y),torch.tensor(y_))
                        loss.requires_grad_(True)
                        loss.backward()
                        self.optimizer_1.step()
                        self.optimizer_1.zero_grad()

                        state, action, reward, next_state, done = self.p2_reply_buffer.choice_one()
                        # 使用这一个来进行梯度下降
                        y_ = reward + self.parameters.gamma * self.p2_target(next_state)[action].item()
                        y = torch.max(self.p2_target(state)).item()
                        loss = self.loss_function_2(torch.tensor(y),torch.tensor(y_))
                        loss.requires_grad_(True)
                        loss.backward()
                        self.optimizer_2.step()
                        self.optimizer_2.zero_grad()

            # 从外部非支配集合来获取数据进行训练

            # 让12阶段的online参数复制target参数
            if epoch % self.parameters.C == 0:
                self.p1_online.load_state_dict(self.p1_target.state_dict())
                self.p2_online.load_state_dict(self.p2_target.state_dict())

            # 所有轮数训练完毕，保存网络参数
            torch.save(self.p1_target.state_dict(), "net/p1_target.pth")
            torch.save(self.p2_target.state_dict(), "net/p2_target.pth")

            print("job_process_end_time:")
            print(self.situation.jobsProcessEndTime)
            print(f"makespan = {np.max(self.situation.jobsProcessEndTime)}")

            print("totalprocess and processToMachine:")
            print(self.situation.totalprocess)
            print(self.situation.processToMachine)

            self.situation.clear()

if __name__ == '__main__':
    r = Run()
    r.start()




























