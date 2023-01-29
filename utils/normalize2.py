import numpy as np
import pickle
import sys
import os

class RunningMeanStd(object):
    def __init__(self, save_dir, state_dim, limit_cnt=1e6):
        self.file_name = f"{save_dir}/normalize.pkl"
        self.limit_cnt = limit_cnt
        if os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.mean, self.var, self.count = pickle.load(f)
        else:
            self.mean = np.zeros(state_dim, np.float32)
            self.var = np.ones(state_dim, np.float32)
            self.count = 0.0

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        return

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if self.count >= self.limit_cnt: return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        return

    def normalize(self, observations):
        return (observations - self.mean)/np.sqrt(self.var + 1e-8)

    def save(self):
        with open(self.file_name, 'wb') as f:
            pickle.dump([self.mean, self.var, self.count], f)
