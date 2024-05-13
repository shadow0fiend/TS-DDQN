import numpy as np
import torch
class Parameters:
    def __init__(self):
        self.init_jobs = np.random.randint(10,20)
        self.insert_jobs = np.random.randint(10,20)
        self.jobs = self.init_jobs + self.insert_jobs
        self.machines = np.random.randint(20,30)
        self.process = np.random.randint(10,15)
        self.error_start = 10
        self.error_machine = 2
        self.error_time = 10
        self.DDT = np.random.random() + 0.5
        self.pr = self.urgency(self.jobs)
        self.A = self.arrive(self.init_jobs,self.insert_jobs)
        self.minibatch_size = 64
        self.epsilon = 0.4
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.C = 20
        self.epochs = 5
        self.capacity_history = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.0003
    def arrive(self,init_job ,insert_job):
        ret = [0 for i in range(init_job)]
        lambdas = np.random.uniform(50, 100, insert_job)  # 从均匀分布中生成 λ
        data = [int(np.random.exponential(scale=1 / lam) * 1000) for lam in lambdas]  # 生成指数分布数据
        [ret.append(data[i]) for i in range(len(data))]
        return ret
    def due_time(self, a, data, process_num, DDT):
        ave = []
        time = 0
        for i in range(len(data)):
            filtered_list = [x for x in data[i] if x != '-']
            mean_value = sum(filtered_list) / len(filtered_list)
            time += mean_value
            if (i % process_num == 0):
                ave.append(time)
                time = 0
        self.D = [a[i] + (ave[i] * DDT) for i in range(len(a))]
    def urgency(self,job_num):
        return [np.random.randint(1,5) for i in range(job_num)]
    def set_loss(self,parameters,learning_rate):
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)
        loss_function = torch.nn.MSELoss()
        return optimizer,loss_function