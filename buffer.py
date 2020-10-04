import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        # self.buffer.append((s0[None, :], a, r, s1[None, :], done))
        self.buffer.append((s0, a, r, s1, done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # arr1 = np.array(s0)
        return np.concatenate(s0), a, r, np.concatenate(s1), done   #(32, 6, 96, 96)

    def size(self):
        return len(self.buffer)