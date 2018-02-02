import torch
import torch.multiprocessing as mp 
import numpy as np

# This is from the https://github.com/alexis-jacq/Pytorch-DPPO/blob/master/utils.py#L9

# this is to make sure if the workers could pass gradient to the chief...
class TrafficLight:
    def __init__(self):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value


    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

# this is used to decide when the chief could update the network...
class Counter:
    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def increment(self):
        with self.lock:
            self.val.value += 1

    def reset(self):
        with self.lock:
            self.val.value = 0

# this is used to record the reward each worker achieved...
class RewardCounter:
    def __init__(self):
        self.val = mp.Value('f', 0)
        self.lock = mp.Lock()

    def add(self, reward):
        with self.lock:
            self.val.value += reward

    def get(self):
        with self.lock:
            return self.val.value

    def reset(self):
        with self.lock:
            self.val.value = 0

# this is used to accumulate the gradients
class Shared_grad_buffers:
    def __init__(self, models):
        self.lock = mp.Lock()
        self.grads = {}
        for name, p in models.named_parameters():
            self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_()

    def add_gradient(self, models):
        with self.lock:
            for name, p in models.named_parameters():
                self.grads[name + '_grad'] += p.grad.data

    def reset(self):
        with self.lock:
            for name, grad in self.grads.items():
                self.grads[name].fill_(0)


# running mean filter, used to normalize the state of mujoco environment
class Running_mean_filter:
    def __init__(self, num_inputs):
        self.lock = mp.Lock()
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.s = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()
    # start to normalize the states...
    def normalize(self, x):
        with self.lock:
            obs = x.copy()
            obs = torch.Tensor(obs)
            self.n += 1
            if self.n[0] == 1:
                self.mean[...] = obs  
                self.var[...] = self.mean.pow(2)
            else:
                old_mean = self.mean.clone()
                self.mean[...] = old_mean + (obs - old_mean) / self.n
                self.s[...] = self.s + (obs - old_mean) * (obs - self.mean)
                self.var[...] = self.s / (self.n - 1)
            mean_clip = self.mean.numpy().copy()
            var_clip = self.var.numpy().copy()
            std = np.sqrt(var_clip)
            x = (x - mean_clip) / (std + 1e-8)
            x = np.clip(x, -5.0, 5.0)
            return x
    # start to get the results...
    def get_results(self):
        with self.lock:
            var_clip = self.var.numpy().copy()
            return (self.mean.numpy().copy(), np.sqrt(var_clip))



