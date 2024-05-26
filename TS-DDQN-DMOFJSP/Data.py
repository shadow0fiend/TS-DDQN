import numpy as np
import pandas as pd
class DataFactory:
    def generate(self,path,job_num,process,machine_num):
        data = np.zeros((job_num * process,machine_num),dtype=object)
        for i in range(job_num * process):
            for j in range(machine_num):
                if np.random.random() < 0.15:
                    data[i,j] = '-'
                else:
                    data[i,j] = np.random.randint(1,10)
        pd.DataFrame(data).to_csv(path, index=False, header=False, sep=' ')

if __name__ == '__main__':
    d = DataFactory()
    for i in range(5):
        d.generate(path=f"../data/0.5-20-50-{i + 1}",job_num=20,process=5,machine_num=20)
