import numpy as np
class ReplayBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = np.random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def choice_one(self):
        index = np.random.randint(0,len(self.buffer))
        return self.buffer[index]
    def __len__(self):
        return len(self.buffer)
