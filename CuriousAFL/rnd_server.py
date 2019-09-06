"""
Implements a RND Network and launches as a Apache Thrift RPC server locally.
"""

from collections import deque
from statistics import median
import numpy as np
import argparse
import random
import sys
import zmq
from torch.utils.tensorboard import SummaryWriter
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

replay_buffer = deque(maxlen=int(BUFFER_SIZE/5))
reward_buffer = deque(maxlen=int(BUFFER_SIZE/5))
rnd_model = None
step_counter = 0
writer = None
device = None  # pytorch device
analysis_step_count = 0


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
        self.target = NN(in_dim, out_dim, n_hid).to(device=device)
        self.model = NN(in_dim, out_dim, n_hid).to(device=device)
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
            print("init rnd")
            rnd_model = RND(in_dim=MAX_FILESIZE, out_dim=1, n_hid=124, lr=self.args.learningrate)

            if self.args.tensorboard:
                global writer
                writer = SummaryWriter()

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

        byte_array = np.fromfile(self.args.projectbase + seed, 'u1')
        byte_array = byte_array / 255

        #byte_array = np.unpackbits(byte_array)  # min max normalized

        if len(byte_array) > MAX_FILESIZE:
            byte_array = byte_array[:MAX_FILESIZE]
        else:
            byte_array = np.pad(byte_array, (0, MAX_FILESIZE - len(byte_array)), 'constant',
                                constant_values=0)

        state = torch.tensor(byte_array, dtype=torch.float).to(device=device)

        reward = rnd_model.get_reward(state).detach().clamp(0.0, 1.0).item()

        if step_counter % 1000 == 0 and self.args.tensorboard:
            global analysis_step_count
            writer.add_scalar('RND reward', reward, analysis_step_count)
            analysis_step_count += 1

        global reward_buffer
        reward_buffer.append(reward)

        if len(reward_buffer) > 100 and \
                reward < np.percentile(np.array(reward_buffer), [50])[0]:
            #reward < median(list(reward_buffer)[-int(len(reward_buffer)):]):
            return 1

        if np.random.random(1)[0] > 0.75:
            global replay_buffer
            replay_buffer.append(byte_array)

        if step_counter > 1000:
            #update model
            replay_buffer_l = np.array(replay_buffer)
            num = len(replay_buffer)
            K = np.min([num, BATCH_SIZE])
            #samples = np.array(random.sample(replay_buffer, K))
            samples = replay_buffer_l[np.random.choice(replay_buffer_l.shape[0], K, replace=False)]
            S0 = torch.tensor(samples, dtype=torch.float).to(device=device)
            Ri = rnd_model.get_reward(S0)
            rnd_model.update(Ri)
            #pool.apply(self.update_model)
            #print('updated model')
            step_counter = 0

        return 0


def get_open_port():
    # This is very ugly, don't copy this, don't look at this, forget this was ever here
    # it serves as a backup, if user doesn't provide a port. It has a race condition and is generally ugly. Just don't.
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    print("Using: " + str(port))
    return port


def main(args):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:"+str(args.port))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("found cuda device...")
    else:
        device = torch.device('cpu')

    if args.tensorboard:
        print("You wanted Tensorboard (TB) - serverside this activates TB's SummaryWriter. To launch TB on your side, "
              "run: \ntensorboard --logdir=runs")

    dispatcher = Dispatcher(args)
    veto = 1
        
    while True:
        message = socket.recv()

        print("Received request: %s" % message)
        
        if message == "init":
            dispatcher.initModel()
        else:
            veto = dispatcher.veto(message)
        socket.send(veto)


def parse_args():
    """Parse command line arguments.
    Returns:
      Parsed argument object.
    """
    parser = argparse.ArgumentParser(description="Launching RND Network as a Thrift RPC")
    parser.add_argument(
        '--projectbase',
        help='update model after x executions',
        default='.')
    parser.add_argument(
        '--tensorboard', help='launch Tensorboard to monitor curiosity values', type=bool, default=False)
    parser.add_argument('--disable-cuda', type=bool, default=False, help='Disable CUDA')
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
    parser.add_argument(
        '--port',
        help='On which port to communicate with AFL.',
        default=get_open_port,
        type=int)
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    sys.exit(main(parsed_args))
