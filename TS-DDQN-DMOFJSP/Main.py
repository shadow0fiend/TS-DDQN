import numpy as np
import torch
import os
from Data import DataFactory
from ReplayBuffer import ReplayBuffer
from Parameters import Parameters
from Network import p1_Net,p2_Net
from Situation import Situation
class Run():
    def __init__(self):
        self.init()
    def init(self):
        self.parameters = Parameters()
        self.p1_reply_buffer = ReplayBuffer(self.parameters.capacity_history)
        self.p2_reply_buffer = ReplayBuffer(self.parameters.capacity_history)
        self.data_factory = DataFactory()
        self.data_factory.generate("./data/t0.txt",self.parameters.jobs,self.parameters.process,self.parameters.machines)
        self.situation = Situation(self.parameters)
        self.p1_online = p1_Net()
        self.p1_target = p1_Net()
        self.p2_online = p2_Net()
        self.p2_target = p2_Net()
        self.optimizer_1, self.loss_function_1 = self.parameters.set_loss(self.p1_target.parameters(), self.parameters.learning_rate)
        self.optimizer_2, self.loss_function_2 = self.parameters.set_loss(self.p2_target.parameters(),self.parameters.learning_rate)
    def start(self):
        ND = []
        for epoch in range(self.parameters.epochs):
            if os.path.exists('./net/p1_target.pth'):
                self.p1_target.load_state_dict(torch.load('./net/p1_target.pth'))
            if os.path.exists('./net/p2_target.pth'):
                self.p2_target.load_state_dict(torch.load('./net/p2_target.pth'))
            for p in range(self.parameters.jobs * self.parameters.process):
                if p == 0:
                    self.situation.initialProcess()
                    continue
                self.situation.getState()
                p1_online_ret = torch.argmax(self.p1_online.forward(self.situation.stateVector)).item()
                p1_target_ret = self.p1_target.forward(self.situation.stateVector)[p1_online_ret].item()
                p2_online_ret = torch.argmax(self.p2_online.forward(self.situation.stateVector)).item()
                p2_target_ret = self.p2_target.forward(self.situation.stateVector)[p2_online_ret].item()
                jobNum, machineNum = self.situation.Action(p2_online_ret,self.parameters.epsilon)
                print(f"epoch:{epoch + 1}, process:{p+1}, total process:{self.parameters.jobs * self.parameters.process}, assigning job:{jobNum+1} to machine:{machineNum+1}")
                old_state = self.situation.stateVector
                self.situation.caculate()
                self.situation.getState()
                new_state = self.situation.stateVector
                reward = self.situation.reward(p1_online_ret,old_state,new_state)
                self.p1_reply_buffer.push(old_state,p1_online_ret,reward,new_state)
                self.p2_reply_buffer.push(old_state,p2_online_ret,reward,new_state)
                if len(self.p1_reply_buffer.buffer) >= self.parameters.minibatch_size:
                    for i in range(self.parameters.minibatch_size):
                        state, action, reward, next_state, done = self.p1_reply_buffer.choice_one()
                        y_ = reward + self.parameters.gamma * self.p1_target(next_state)[action].item()
                        y = torch.max(self.p1_target(state)).item()
                        loss = self.loss_function_1(torch.tensor(y),torch.tensor(y_))
                        loss.requires_grad_(True)
                        loss.backward()
                        self.optimizer_1.step()
                        self.optimizer_1.zero_grad()
                        state, action, reward, next_state, done = self.p2_reply_buffer.choice_one()
                        y_ = reward + self.parameters.gamma * self.p2_target(next_state)[action].item()
                        y = torch.max(self.p2_target(state)).item()
                        loss = self.loss_function_2(torch.tensor(y),torch.tensor(y_))
                        loss.requires_grad_(True)
                        loss.backward()
                        self.optimizer_2.step()
                        self.optimizer_2.zero_grad()
            if ND == None:
                ND.append([self.situation.totalprocess, self.situation.processToMachine, self.situation.state1, self.situation.state6])
            else:
                tmp = [self.situation.totalprocess, self.situation.processToMachine, self.situation.state1,self.situation.state6]
                for i in range(len(ND)):
                    if tmp[2] > ND[i][2] and tmp[3] < ND[i][3]:
                        ND.pop(i)
                        ND.append(tmp)
            if epoch % self.parameters.C == 0:
                self.p1_online.load_state_dict(self.p1_target.state_dict())
                self.p2_online.load_state_dict(self.p2_target.state_dict())
            torch.save(self.p1_target.state_dict(), "../net/p1_target.pth")
            torch.save(self.p2_target.state_dict(), "../net/p2_target.pth")
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