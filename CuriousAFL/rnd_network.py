"""
RND network to veto AFL executions
"""

#from .rnd import RND
import numpy as np
import random
from collections import deque
import torch
import torch.nn
import torch.nn.functional as F
import sys
from statistics import median

# epochs are controlled by AFL

max_filesize = 2**12
learning_rate = 1e-4

rnd_model = None
buffer_size = 2**10
batch_size = 10**3 # neuzz reference
replay_buffer = deque(maxlen=buffer_size)
reward_buffer = deque(maxlen=int(buffer_size/10))

input_dim = max_filesize
output_dim = 1  # fuzz or don't - 2 possible actions

step_counter = 0

def init_models():
    global rnd_model
    rnd_model = RND(in_dim=max_filesize, out_dim=1, n_hid=124)
    global reward_buffer
    reward_buffer.append(len(reward_buffer) * [0.0])

def rnd_pass(input_file, seed):
    """
    main func for AFL to call
    :param input_file:
    :return: true if seed should be executed
    """

    if rnd_model is None:
        init_models()

    global step_counter
    step_counter += 1

    #if step_counter > retrain_every_x_seeds:
    #    update_model()
    #    step_counter = 0
    # convert file into byte-array
    byte_array = np.fromfile(seed, 'u1')
    byte_array = byte_array / 255 # min max normalized

    if len(byte_array) > max_filesize:
        byte_array = byte_array[:max_filesize]
    else:
        byte_array = np.pad(byte_array, (0, max_filesize - len(byte_array)), 'constant',
                            constant_values=0)


    state = torch.Tensor(byte_array)
    reward = rnd_model.get_reward(state).detach().clamp(0.0, 1.0).item()

    global reward_buffer
    reward_buffer.append(reward)
    
    if len(reward_buffer) < 10 or (reward < median(list(reward_buffer)[-int(len(reward_buffer)/4):])):
        return False


    # we now execute the seed
    global replay_buffer
    replay_buffer.append(byte_array)


    if step_counter > batch_size:
        #update model
        num = len(replay_buffer)
        K = np.min([num, batch_size])
        samples = random.sample(replay_buffer, K)
        print(len(samples))

        S0 = torch.tensor(samples, dtype=torch.float)

        Ri = rnd_model.get_reward(S0)
        rnd_model.update(Ri)

        print('updated model')
        step_counter = 0

    return True


class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = torch.nn.Linear(in_dim, n_hid, 'linear')
        self.fc2 = torch.nn.Linear(n_hid, n_hid, 'linear')
        self.fc3 = torch.nn.Linear(n_hid, out_dim, 'linear')
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        # y = self.softmax(y)
        return y


class RND:
    def __init__(self, in_dim, out_dim, n_hid):
        self.target = NN(in_dim, out_dim, n_hid)
        self.model = NN(in_dim, out_dim, n_hid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_reward(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = torch.pow(y_pred - y_true, 2).mean()
        return reward

    def update(self, Ri):
        Ri.sum().backward()
        self.optimizer.step()
