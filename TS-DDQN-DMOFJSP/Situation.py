import torch
import numpy as np
import Parameters


class Situation:
    def __init__(self, parameters: Parameters):
        # 基础数据
        self.jobs = parameters.jobs
        self.machines = parameters.machines
        self.process = parameters.process

        self.error_start = parameters.error_start
        self.error_machine = parameters.error_machine
        self.error_time = parameters.error_time

        self.A = parameters.A
        self.pr = parameters.pr
        self.DDT = parameters.DDT

        self.getdata("../data/t0.txt")
        parameters.due_time(self.A,self.data,self.process,self.DDT)
        self.D = parameters.D

        # 工序加工信息
        self.totalprocess = []
        self.processToMachine = []
        self.iniCompleteProcess = [i for i in range(self.jobs) for j in range(self.process)]

    # 清理数据
    def clear(self):
        # 工序加工信息
        self.totalprocess = []
        self.processToMachine = []
        self.iniCompleteProcess = [i for i in range(self.jobs) for j in range(self.process)]

    # 随机第一步操作
    def initialProcess(self):
        index = np.random.randint(0, len(self.iniCompleteProcess))
        jobIndex = self.iniCompleteProcess[index]
        self.totalprocess.append(jobIndex)
        self.iniCompleteProcess.pop(index)
        machinesIndex = np.random.randint(0, self.machines)
        while self.data[jobIndex * self.process][machinesIndex] == -1:
            machinesIndex = np.random.randint(0, self.machines)
        self.processToMachine.append(machinesIndex)
        # 更新完工时间信息表
        self.caculate()

    # 加载数据集
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

    # 解码
    def handle(self, x):
        self.array = []  # 工件对应工序
        # x:总工序(0-9)
        # 返回值(工序数:(1-6)，工件序号:(0-9))
        piece_mark = np.zeros(self.jobs, dtype=np.int32)
        for i in range(len(x)):
            piece_mark[x[i]] += 1
            self.array.append((piece_mark[x[i]], x[i]))

    # 计算并更新车间加工信息
    def caculate(self, type=True):
        self.handle(self.totalprocess)
        self.jobsProcessStart = np.zeros(self.jobs * self.process, dtype=np.int32)  # 工序开始时间
        self.jobsProcessWorkTime = np.zeros(self.jobs * self.process, dtype=np.int32)  # 工序加工时间
        self.jobsProcessEndTime = np.zeros((self.jobs, self.process), dtype=np.int32)  # 工序完工时间
        self.machinesEndTime = np.zeros(self.machines, dtype=np.int32)  # 机器完工时间
        self.machineWorkTime = np.zeros(self.machines, dtype=np.int32)  # 机器加工时间
        for ii, i in enumerate(self.array):     # 遍历已完工列表
            process_index = i[1] * self.process + i[0] - 1  # 确定data中第几行
            machine_index = self.processToMachine[ii]  # 确定data中第几列
            while self.data[process_index][machine_index] == -1:
                machine_index = np.random.randint(0, self.machines)
                self.processToMachine[ii] = machine_index
            process_time = self.data[process_index][machine_index]  # 确定加工时间
            self.machineWorkTime[machine_index] += process_time
            if i[0] == 1:  # 如果是第一道工序
                self.jobsProcessStart[ii] = max(0, self.machinesEndTime[machine_index])  # max(0,机器上一次加工完成时间)
                self.jobsProcessWorkTime[ii] = process_time  # 工件工序的加工时间
                if type:  # type=true，表示求的是机器故障的模型，type=false，表示求得是普通的柔性车间模型
                    # 判断：1.如果工序开始时间在故障期间，则改为故障结束时间
                    if machine_index == self.error_machine:
                        # 如果该工序的开始时间大于故障开始时间，并且工序开始时间小于故障结束时间，则将该工序的开始时间改为机器故障恢复时间
                        if self.jobsProcessStart[ii] >= self.error_start and self.jobsProcessStart[ii] <= self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                        # 又或者该工序的完工时间大于机器故障的开始时间，表示在该工序加工的过程中，就已经发生故障了
                        elif self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] > self.error_start:
                            # 则工序的开始时间改为机器的完工时间
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                self.jobsProcessEndTime[i[1], i[0] - 1] = self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii]  # 工件对应工序的完工时间
                self.machinesEndTime[machine_index] = self.jobsProcessEndTime[i[1], i[0] - 1]  # 机器上一次完成时间+加工时间
            else:  # 如果不是第一道工序
                self.jobsProcessStart[ii] = max(self.machinesEndTime[machine_index],self.jobsProcessEndTime[i[1], i[0] - 2])  # max(机器上一次加工完成时间,上一道工序的完工时间)
                self.jobsProcessWorkTime[ii] = process_time
                if type:  # type=true，表示求的是机器故障的模型，type=false，表示求得是普通的柔性车间模型
                    # 判断：1.如果工序开始时间在故障期间，则改为故障结束时间
                    if machine_index == self.error_machine:
                        if self.jobsProcessStart[ii] >= self.error_start and self.jobsProcessStart[ii] <= self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                        elif self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] > self.error_start and self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii] < self.error_start + self.error_time:
                            self.jobsProcessStart[ii] = self.error_start + self.error_time
                self.jobsProcessEndTime[i[1], i[0] - 1] = self.jobsProcessStart[ii] + self.jobsProcessWorkTime[ii]  # 工件对应工序的完工时间
                self.machinesEndTime[machine_index] = self.jobsProcessEndTime[i[1], i[0] - 1]

    # 假设机器发生故障
    # 排查故障发生的时候，各工序加工状态如何
    def findError(self, jobsProcessStart, jobsProcessWorkTime, totalprocess, processToMachine):
        machineLastEndTime = 0
        for i in range(len(totalprocess)):  # 遍历总的工序
            if processToMachine[i] == self.error_machine:  # 如果该工序对应的机器是故障机器，则需要进行判断
                # 如果故障开始时间发生在机器的空闲区域（上次结束 到 下次开始），并且故障机器的持续时间超过了这段空闲时间
                if self.error_start >= machineLastEndTime and self.error_start <= jobsProcessStart[i]:
                    findindex = i  # 则表示找到了，正是在当前工序出现故障
                    return findindex
                # 如果故障发生的时间，恰好在当前工序的加工时间中
                elif self.error_start >= jobsProcessStart[i] and self.error_start <= jobsProcessStart[i] + \
                        jobsProcessWorkTime[i]:
                    findindex = i  # 则表示找到了，正是在当前工序出现故障
                    return findindex
                else:  # 如果不满足，则表示要么故障时间在机器空闲区域之中，可以不用更改
                    machineLastEndTime = jobsProcessStart[i] + jobsProcessWorkTime[i]

    # 计算各个状态的值
    def State12(self):
        # 状态1：机器的平均利用率
        # 状态2：机器利用率的标准偏差
        # 每一台机器的平均利用率 = 每一台机器的工作时间 / 每一台机器的完工时间
        temp = [0 for i in range(self.machines)]
        for i in range(self.machines):
            if self.machinesEndTime[i] == 0:
                temp[i] = 0
            else:
                temp[i] = self.machineWorkTime[i] / self.machinesEndTime[i]
        self.state1 = np.mean(temp)
        self.state2 = np.sqrt(np.sum([((i - self.state1) ** 2) for i in temp]) / self.machines)

    def State34(self):
        # 状态3：工件的平均完成率
        # 状态4：工件完工率的标准偏差
        # 工件的平均完成率 = 每个工件的完成率之和 / 总的工件数
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
        # 状态5：估计延期率
        T_cur = np.mean(self.machinesEndTime)  # 机器完工时间平均值
        N_tard = 0
        N_left = 0
        # 遍历每一个工件
        for i in range(self.jobs):
            # 如果该工件还有没做完的工序
            T_left = 0  # 剩余工序的预测加工时间
            nonCompleteProcess = self.process - np.count_nonzero(self.jobsProcessEndTime[i])
            if nonCompleteProcess > 0:
                N_left += nonCompleteProcess  # 未做完的工序数加起来
            # 遍历每一个未完成的工序
            for j in range(len(self.jobsProcessEndTime[i])):
                if self.jobsProcessEndTime[i][j] == 0:
                    # 计算该工件该工序在所有机器上的平均加工时间
                    avgTempWorkTime = np.mean([i if i > 0 else 0 for i in self.data[i * self.process + j]])
                    T_left += avgTempWorkTime  # 表示预测的，未完成的工序累加的值
                    if T_cur + T_left > self.D[i]:  # 如果当前已经完成的时间+未完成的时间超过了交期
                        N_tard += self.process - j + 1  # 那么就认为该工件的剩余工序数量是延期的
                        break
        if N_left == 0: # 如果未完成的工序数量是0，即所有工序都完成了，那么state5 = 0
            self.state5 = 0
        else:
            self.state5 = N_tard / N_left  # 最终判断 可能延期的工序数量 / 总的未完成的工序数量

    def State6(self):
        N_tard = 0
        N_left = 0
        # 遍历每一个工件
        for i in range(self.jobs):
            # 如果该工件有未完成的工序
            nonCompleteProcess = self.jobs - np.count_nonzero(self.jobsProcessEndTime[i])
            if nonCompleteProcess > 0:
                N_left += self.process - np.count_nonzero(self.jobsProcessEndTime[i])
                # 如果该工件的某个工序的完工时间大于交期
                for j in range(self.process):
                    if self.jobsProcessEndTime[i][j] > self.D[i]:
                        # 那么就把剩下的工序都加起来，因为他们也都是延期的
                        N_tard += self.process - np.count_nonzero(self.jobsProcessEndTime[i])
                        break
        # 延期的工序/所有未完成的工序
        if N_left == 0:
            self.state6 = 0
        else:
            self.state6 = N_tard / N_left

    # 把环境拼接成一个向量
    def getState(self):
        self.State12(), self.State34(), self.State5(), self.State6()
        self.stateVector = torch.tensor([self.state1, self.state2, self.state3, self.state4, self.state5, self.state6])

    # 定义行为action，action的作用是选择工件+指定机器
    def Action1(self):
        T_cur = np.mean(self.machinesEndTime)
        Tard_job = []
        UC_job = []
        # 先找出可能会延期的工件->有工序未完成 并且 max(T_cur,完工时间) > D[i]
        for i in range(len(self.jobsProcessEndTime)):
            # 如果该工件还有工序没有完成
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
                # 并且max(T_cur,MakeSpan)
                for j in range(len(self.jobsProcessEndTime[i])):
                    if max(T_cur, self.jobsProcessEndTime[i][j]) > self.D[i]:
                        Tard_job.append(i)
        # 如果Tard_job是空的
        if len(Tard_job) == 0:
            # 选择工件
            tempList = []
            for i in UC_job:
                tempList.append((self.D[i] - max(T_cur, np.max(self.jobsProcessEndTime[i]))) / ((self.process - np.count_nonzero(self.jobsProcessEndTime[i])) * self.pr[i]))
            # 找到工件了
            jobsNum = UC_job[np.argmin(tempList)]
        else:
            tempList = []
            # 遍历 Tard_job中的每一个工件
            for i in Tard_job:
                # 每个工件未完成工序的平均加工时间
                avgRemainWorkTime = 0
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        # 计算该工件该工序在所有机器上的平均加工时间
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]) * self.pr[i])
            jobsNum = Tard_job[np.argmin(tempList)]
        # 选机器
        # 机器序号为遍历每一台机器，从中选择max(机器的完工时间，当前工件的上一道工序完工时间，工件到达时间)最小的机器。
        machineNum = np.argmin([max(i, np.max(self.jobsProcessEndTime[jobsNum]), self.A[jobsNum]) for i in self.machinesEndTime])
        return jobsNum, machineNum

    def Action2(self):
        T_cur = np.mean(self.machinesEndTime)
        Tard_job = []
        UC_job = []
        # 先找出可能会延期的工件->有工序未完成 并且 max(T_cur,完工时间) > D[i]
        for i in range(len(self.jobsProcessEndTime)):
            # 如果该工件还有工序没有完成
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
                # 并且max(T_cur,MakeSpan)
                for j in range(len(self.jobsProcessEndTime[i])):
                    if max(T_cur, self.jobsProcessEndTime[i][j]) > self.D[i]:
                        Tard_job.append(i)
        # 如果Tard_job是空的
        if len(Tard_job) == 0:
            # 选择工件
            tempList = []
            for i in UC_job:
                avgRemainWorkTime = 0
                # 遍历这个工件
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        # 计算该工件后续未完成的工序在所有机器上的平均加工时间
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (self.D[i] - max(T_cur, np.max(self.jobsProcessEndTime[i]))) / ((avgRemainWorkTime) * self.pr[i]))
            # 找到工件了
            jobsNum = UC_job[np.argmin(tempList)]
        else:
            tempList = []
            # 遍历 Tard_job中的每一个工件
            for i in Tard_job:
                # 每个工件未完成工序的平均加工时间
                avgRemainWorkTime = 0
                for j in range(len(self.jobsProcessEndTime[i])):
                    if self.jobsProcessEndTime[i][j] == 0:
                        # 计算该工件该工序在所有机器上的平均加工时间
                        avgRemainWorkTime += np.mean(self.data[i * self.process + j])
                tempList.append(
                    (max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]) * self.pr[i])
            jobsNum = Tard_job[np.argmin(tempList)]
        # 选机器
        machineNum = np.argmin(
            [max(i, np.max(self.jobsProcessEndTime[jobsNum]), self.A[jobsNum]) for i in self.machinesEndTime])
        return jobsNum, machineNum

    def Action3(self):
        T_cur = np.mean(self.machinesEndTime)
        UC_job = []
        for i in range(self.jobs):  # 遍历每一个工件
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process: # 如果工件有未完成的工序
                UC_job.append(i)    # 就把这个工件加入到UC_job中
        temp = []
        for i in UC_job:
            # 每个工件未完成工序的平均加工时间
            avgRemainWorkTime = 0
            for j in range(len(self.jobsProcessEndTime[i])):
                if self.jobsProcessEndTime[i][j] == 0:
                    # 计算该工件该工序在所有机器上的平均加工时间，计算平均加工时间，不能用有-1的
                    avgRemainWorkTime += np.mean([i if i > 0 else 0 for i in self.data[i * self.process + j]])
            t1 = max(T_cur, np.max(self.jobsProcessEndTime[i])) + avgRemainWorkTime - self.D[i]
            if t1 < 0:
                temp.append(t1)
            else:
                temp.append(t1 * self.pr[i])
        jobNum = UC_job[np.argmax(temp)]
        if np.random.random() < 0.5:
            # 机器负荷最小的，即机器加工时间 / 机器完工时间
            machineNum = np.argmin([self.machineWorkTime[i] / self.machinesEndTime[i] for i in range(self.machines)])
        else:
            # 选择机器加工时间最小的
            machineNum = np.argmin(self.machineWorkTime)
        return jobNum, machineNum

    def Action4(self):
        # 随机选择一个未完成工序
        UC_job = []
        for i in range(len(self.jobsProcessEndTime)):
            if np.count_nonzero(self.jobsProcessEndTime[i]) < self.process:
                UC_job.append(i)
        # 工序为
        jobNum = UC_job[np.random.randint(0, len(UC_job))]
        # 机器为
        machineNum = np.argmin([max(i, np.max(self.jobsProcessEndTime[jobNum]), self.A[jobNum]) for i in self.machinesEndTime])
        return jobNum, machineNum

    # 更新完工序列
    def update(self, jobNum, machineNum):
        self.totalprocess.append(jobNum)
        self.processToMachine.append(machineNum)
        self.iniCompleteProcess.remove(jobNum)

    def Action(self, actionIndex,epsilon):
        if np.random.random() < epsilon:
            # 随机选择机器和工件
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

    # 根据state和next_state来计算reward值
    # reward使用机器平均利用率和预估延期率来比较
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

    # 非支配排序
    def div(self, solve: list):
        n, s, res, temp1 = [], [], [], []
        for p in solve:
            count = 0
            temp2 = []
            for q in solve:
                if p == q:
                    continue
                if p[0] <= q[0] and p[1] <= q[1]:
                    temp2.append(q)
                elif p[0] >= q[0] and p[1] >= q[1]:
                    count += 1
            s.append(temp2)
            n.append(count)
            if count == 0:
                temp1.append(p)
        res.append(temp1)
        i = 0
        while True:
            h = []
            for p in res[i]:
                for q in s[solve.index(p)]:
                    idx = solve.index(q)
                    n[idx] -= 1
                    if n[idx] == 0:
                        h.append(q)
            if not h:
                break
            i += 1
            res.append(h)
        return res

