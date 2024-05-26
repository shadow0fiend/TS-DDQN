import torch
import numpy as np
import Parameters
class Situation:
    def __init__(self, parameters: Parameters):
        self.jobs = parameters.jobs
        self.machines = parameters.machines
        self.process = parameters.process
        self.error_start = parameters.error_start
        self.error_machine = parameters.error_machine
        self.error_time = parameters.error_time
        self.A = parameters.A
        self.pr = parameters.pr
        self.DDT = parameters.DDT
        self.getdata("./data/t0.txt")
        parameters.due_time(self.A,self.data,self.process,self.DDT)
        self.D = parameters.D
        self.totalprocess = []
        self.processToMachine = []
        self.iniCompleteProcess = [i for i in range(self.jobs) for j in range(self.process)]
    def clear(self):
        self.totalprocess = []
        self.processToMachine = []
        self.iniCompleteProcess = [i for i in range(self.jobs) for j in range(self.process)]
    def initialProcess(self):
        index = np.random.randint(0, len(self.iniCompleteProcess))
        jobIndex = self.iniCompleteProcess[index]
        self.totalprocess.append(jobIndex)
        self.iniCompleteProcess.pop(index)
        machinesIndex = np.random.randint(0, self.machines)
        while self.data[jobIndex * self.process][machinesIndex] == -1:
            machinesIndex = np.random.randint(0, self.machines)
        self.processToMachine.append(machinesIndex)
        self.caculate()
    def getdata(self, path):
        data = []
        with open(path, mode="r") as f:
            string = f.readlines()
            for item in string:
                data.append(item.strip().split(" "))
            data_result = []
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] != "-":
                        data_result.append(int(data[i][j]))
                    else:
                        data_result.append(-1)
            self.data = np.array(data_result).reshape(self.jobs * self.process, self.machines)
        return self.data
    def handle(self, x):
        self.array = []
        piece_mark = np.zeros(self.jobs, dtype=np.int32)
        for i in range(len(x)):
            piece_mark[x[i]] += 1
            self.array.append((piece_mark[x[i]], x[i]))
    def caculate(self, type=True):
        self.handle(self.totalprocess)
        self.jobsProcessStart = np.zeros(self.jobs * self.process, dtype=np.int32)
        self.jobsProcessWorkTime = np.zeros(self.jobs * self.process, dtype=np.int32)
        self.jobsProcessEndTime = np.zeros((self.jobs, self.process), dtype=np.int32)
        self.machinesEndTime = np.zeros(self.machines, dtype=np.int32)
        self.machineWorkTime = np.zeros(self.machines, dtype=np.int32)
        for ii, i in enumerate(self.array):
            process_index = i[1] * self.process + i[0] - 1
            machine_index = self.processToMachine[ii]
            while self.data[process_index][machine_index] == -1:
                machine_index = np.random.randint(0, self.machines)
                self.processToMachine[ii] = machine_index
            process_time = self.data[process_index][machine_index]
            self.machineWorkTime[machine_index] += process_time
            if i[0] == 1:
                self.jobsProcessStart[ii] = max(0, self.machinesEndTime[machine_index])
                self.jobsProcessWorkTime[ii] = process_time
                if type:
                    if machine_index == self.error_machine:
                        if self.jobsProcessStart[ii] >= self.error_start and self.jobsProcessStart[ii] <= self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                        elif self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] > self.error_start:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                self.jobsProcessEndTime[i[1], i[0] - 1] = self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii]
                self.machinesEndTime[machine_index] = self.jobsProcessEndTime[i[1], i[0] - 1]
            else:
                self.jobsProcessStart[ii] = max(self.machinesEndTime[machine_index],self.jobsProcessEndTime[i[1], i[0] - 2])
                self.jobsProcessWorkTime[ii] = process_time
                if type:
                    if machine_index == self.error_machine:
                        if self.jobsProcessStart[ii] >= self.error_start and self.jobsProcessStart[ii] <= self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                        elif self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] > self.error_start and self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] < self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                self.jobsProcessEndTime[i[1], i[0] - 1] = self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii]
                self.machinesEndTime[machine_index] = self.jobsProcessEndTime[i[1], i[0] - 1]
    def findError(self, jobsProcessStart, jobsProcessWorkTime, totalprocess, processToMachine):
        machineLastEndTime = 0
        for i in range(len(totalprocess)):
            if processToMachine[i] == self.error_machine:
                if self.error_start >= machineLastEndTime and self.error_start <= jobsProcessStart[i]:
                    findindex = i
                    return findindex
                elif self.error_start >= jobsProcessStart[i] and self.error_start <= jobsProcessStart[i] + \
                        jobsProcessWorkTime[i]:
                    findindex = i
                    return findindex
                else:
                    machineLastEndTime = jobsProcessStart[i] + jobsProcessWorkTime[i]
    def State12(self):
        temp = [0 for i in range(self.machines)]
        for i in range(self.machines):
            if self.machinesEndTime[i] == 0:
                temp[i] = 0
            else:
                temp[i] = self.machineWorkTime[i] / self.machinesEndTime[i]
        self.state1 = np.mean(temp)
        self.state2 = np.sqrt(np.sum([((i - self.state1) ** 2) for i in temp]) / self.machines)

    def State34(self):
        tempAvg = [0 for i in range(self.jobs)]
        for i in range(self.jobs):
            count = 0
            for j in range(self.process):
                if self.jobsProcessEndTime[i][j] > 0:
                    count += 1
            tempAvg[i] = count / self.process
        self.state3 = np.sum(tempAvg) / self.jobs
        self.state4 = np.sqrt(np.sum([((i - self.state3) ** 2) for i in tempAvg]) / self.jobs)

    def State5(self):
        T_cur = np.mean(self.machinesEndTime)
        N_tard = 0
        N_left = 0
        for i in range(self.jobs):
            T_left = 0
            nonCompleteProcess = self.process - np.count_nonzero(self.jobsProcessEndTime[i])
            if nonCompleteProcess > 0:
                N_left += nonCompleteProcess
            for j in range(len(self.jobsProcessEndTime[i])):
                if self.jobsProcessEndTime[i][j] == 0:
                    avgTempWorkTime = np.mean([i if i > 0 else 0 for i in self.data[i * self.process + j]])
                    T_left += avgTempWorkTime
                    if T_cur + T_left > self.D[i]:
                        N_tard += self.process - j + 1
                        break
        if N_left == 0:
            self.state5 = 0
        else:
            self.state5 = N_tard / N_left

    def State6(self):
        N_tard = 0
        N_left = 0
        for i in range(self.jobs):
            nonCompleteProcess = self.jobs - np.count_nonzero(self.jobsProcessEndTime[i])
            if nonCompleteProcess > 0:
                N_left += self.process - np.count_nonzero(self.jobsProcessEndTime[i])
                for j in range(self.process):
                    if self.jobsProcessEndTime[i][j] > self.D[i]:
                        N_tard += self.process - np.count_nonzero(self.jobsProcessEndTime[i])
                        break
        if N_left == 0:
            self.state6 = 0
        else:
            self.state6 = N_tard / N_left
    def getState(self):
        self.State12(), self.State34(), self.State5(), self.State6()
        self.stateVector = torch.tensor([self.state1, self.state2, self.state3, self.state4, self.state5, self.state6])
    def Action1(self):
        T_cur = np.mean(self.machinesEndTime)
        Tard_job = []
        UC_job = []
        for i in range(len(self.jobsProcessEndTime)):
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
                for j in range(len(self.jobsProcessEndTime[i])):
                    if max(T_cur, self.jobsProcessEndTime[i][j]) > self.D[i]:
                        Tard_job.append(i)
        if len(Tard_job) == 0:
            tempList = []
            for i in UC_job:
                tempList.append((self.D[i] - max(T_cur, np.max(self.jobsProcessEndTime[i]))) / ((self.process - np.count_nonzero(self.jobsProcessEndTime[i])) * self.pr[i]))
            jobsNum = UC_job[np.argmin(tempList)]
        else:
            tempList = []
            for i in Tard_job:
                avgRemainWorkTime = 0
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]) * self.pr[i])
            jobsNum = Tard_job[np.argmin(tempList)]
        machineNum = np.argmin([max(i, np.max(self.jobsProcessEndTime[jobsNum]), self.A[jobsNum]) for i in self.machinesEndTime])
        return jobsNum, machineNum

    def Action2(self):
        T_cur = np.mean(self.machinesEndTime)
        Tard_job = []
        UC_job = []
        for i in range(len(self.jobsProcessEndTime)):
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
                for j in range(len(self.jobsProcessEndTime[i])):
                    if max(T_cur, self.jobsProcessEndTime[i][j]) > self.D[i]:
                        Tard_job.append(i)
        if len(Tard_job) == 0:
            tempList = []
            for i in UC_job:
                avgRemainWorkTime = 0
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (self.D[i] - max(T_cur, np.max(self.jobsProcessEndTime[i]))) / ((avgRemainWorkTime) * self.pr[i]))
            jobsNum = UC_job[np.argmin(tempList)]
        else:
            tempList = []
            for i in Tard_job:
                avgRemainWorkTime = 0
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]) * self.pr[i])
            jobsNum = Tard_job[np.argmin(tempList)]
        machineNum = np.argmin(
            [max(i, np.max(self.jobsProcessEndTime[jobsNum]), self.A[jobsNum]) for i in self.machinesEndTime])
        return jobsNum, machineNum

    def Action3(self):
        T_cur = np.mean(self.machinesEndTime)
        UC_job = []
        for i in range(self.jobs):
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process: # 如果工件有未完成的工序
                UC_job.append(i)
        temp = []
        for i in UC_job:
            avgRemainWorkTime = 0
            for j in range(len(self.jobsProcessEndTime[i])):
                if self.jobsProcessEndTime[i][j] == 0:
                    avgRemainWorkTime += np.mean([i if i > 0 else 0 for i in self.data[i * self.process + j]])
            t1 = max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]
            if t1 < 0:
                temp.append(t1)
            else:
                temp.append(t1 * self.pr[i])
        jobNum = UC_job[np.argmax(temp)]
        if np.random.random() < 0.5:
            machineNum = np.argmin([self.machineWorkTime[i] / self.machinesEndTime[i] for i in range(self.machines)])
        else:
            machineNum = np.argmin(self.machineWorkTime)
        return jobNum, machineNum

    def Action4(self):
        UC_job = []
        for i in range(len(self.jobsProcessEndTime)):
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
        jobNum = UC_job[np.random.randint(0, len(UC_job))]
        machineNum = np.argmin([max(i, np.max(self.jobsProcessEndTime[jobNum]), self.A[jobNum]) for i in self.machinesEndTime])
        return jobNum, machineNum
    def update(self, jobNum, machineNum):
        self.totalprocess.append(jobNum)
        self.processToMachine.append(machineNum)
        self.iniCompleteProcess.remove(jobNum)

    def Action(self, actionIndex,epsilon):
        if np.random.random() < epsilon:
            jobNum = np.random.choice(self.iniCompleteProcess)
            machineNum = np.random.randint(0,self.machines)
            self.update(jobNum,machineNum)
            return jobNum,machineNum
        if actionIndex == 0:
            jobNum, machineNum = self.Action1()
            self.update(jobNum, machineNum)
            return jobNum,machineNum
        elif actionIndex == 1:
            jobNum, machineNum = self.Action2()
            self.update(jobNum, machineNum)
            return jobNum,machineNum
        elif actionIndex == 2:
            jobNum, machineNum = self.Action3()
            self.update(jobNum, machineNum)
            return jobNum,machineNum
        elif actionIndex == 3:
            jobNum, machineNum = self.Action4()
            self.update(jobNum, machineNum)
            return jobNum,machineNum

    def reward(self, goal, state, next_state):
        if goal == 0:
            if next_state[5] < state[5]:
                return 1
            elif next_state[5] > state[5]:
                return -1
            else:
                return 0
        if goal == 1:
            if next_state[0] > state[0]:
                return 1
            elif next_state[0] > state[0] * 0.95:
                return 0
            else:
                return -1