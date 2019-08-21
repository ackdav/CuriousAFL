"""
Implements a RND Network and launches as a Apache Thrift RPC server locally.
"""

from collections import deque
from statistics import median
import numpy as np
import argparse
import random
import sys

import thriftpy2
from thriftpy2.rpc import make_server
import torch
import torch.nn
import torch.nn.functional as F

# RND constants - TODO: optimize
MAX_FILESIZE = 2 ** 12
LEARNING_RATE = 1e-4
BUFFER_SIZE = 2 ** 10  # how many seeds to keep in memory
BATCH_SIZE = 10 ** 4  # update reference model after X executions
INPUT_DIM = MAX_FILESIZE  # input dimension of RND
OUTPUT_DIM = 1  # output dimension of RND


replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque(maxlen=int(BUFFER_SIZE / 10))
rnd_model = None
step_counter = 0


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
    def __init__(self, in_dim, out_dim, n_hid, lr):
        self.target = NN(in_dim, out_dim, n_hid)
        self.model = NN(in_dim, out_dim, n_hid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_reward(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = torch.pow(y_pred - y_true, 2).mean()
        return reward

    def update(self, Ri):
        Ri.sum().backward()
        self.optimizer.step()


class Dispatcher(object):
    def __init__(self, args):
        self.args = args
        self.log = {}

    def initModel(self):
        try:
            global rnd_model
            rnd_model = RND(in_dim=MAX_FILESIZE, out_dim=1, n_hid=124, lr=self.args.learningrate)

            global reward_buffer
            reward_buffer.append(len(reward_buffer) * [0.0])

            print("model&buffer ready")
            return 0
        except:
            return 1

    def veto(self, seed):
        """
        main func for AFL to call
        :param seed:
        :return: 0 if seed should be executed, 1 if should be skipped
        """

        global step_counter
        step_counter += 1

        byte_array = np.fromfile(seed, 'u1')
        byte_array = byte_array / 255  # min max normalized

        if len(byte_array) > MAX_FILESIZE:
            byte_array = byte_array[:MAX_FILESIZE]
        else:
            byte_array = np.pad(byte_array, (0, MAX_FILESIZE - len(byte_array)), 'constant',
                                constant_values=0)

        state = torch.Tensor(byte_array)
        reward = rnd_model.get_reward(state).detach().clamp(0.0, 1.0).item()

        global reward_buffer
        reward_buffer.append(reward)

        global replay_buffer
        replay_buffer.append(byte_array)

        if len(reward_buffer) < 10 or (reward < median(list(reward_buffer)[-int(len(reward_buffer) / 4):])):
            return 1

        if step_counter > BATCH_SIZE:
            # update model
            num = len(replay_buffer)
            K = np.min([num, BATCH_SIZE])
            samples = random.sample(replay_buffer, K)

            S0 = torch.tensor(samples, dtype=torch.float)

            Ri = rnd_model.get_reward(S0)
            rnd_model.update(Ri)

            print('updated model')
            step_counter = 0

        return 0


def main(args):
    rnd_thrift = thriftpy2.load("rnd.thrift", module_name="rnd_thrift")
    server = make_server(rnd_thrift.Rnd, Dispatcher(args), '127.0.0.1', 6000, client_timeout=None)
    print("serving...")
    server.serve()

def parse_args():
    """Parse command line arguments.
    Returns:
      Parsed arguement object.
    """
    parser = argparse.ArgumentParser(description="Launching RND Network as a Thrift RPC")
    parser.add_argument(
        '--tensorboard', help='launch Tensorboard to monitor curiosity values', type=bool, default=False)
    parser.add_argument(
        '--batchsize',
        help='update model after x executions',
        type=int,
        default=BATCH_SIZE)
    parser.add_argument(
        '--learningrate',
        help='learning rate of predictor model',
        default=LEARNING_RATE)
    parser.add_argument(
        '--inputdim',
        help='control the max considered input size of seed, also controls input of RND.',
        default=INPUT_DIM)
    parser.add_argument(
        '--outputdim',
        help='Controls output dim of RND.',
        default=OUTPUT_DIM)
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    sys.exit(main(parsed_args))
