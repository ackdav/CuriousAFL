"""
RND network to veto AFL executions
"""

from rnd import RND
import numpy as np
import random
from collections import deque
import torch

# epochs are controlled by AFL
retrain_every_x_seeds = 10 ** 4

max_filesize = 2 ** 14
learning_rate = 1e-4

rnd_model = None
buffer_size = 64
replay_buffer = deque(maxlen=buffer_size)

input_dim = max_filesize
output_dim = 2  # fuzz or don't - 2 possible actions

batch_size = 64  # TODO: WHY?

step_counter = 0


def init_models():
    global rnd_model
    rnd_model = RND(in_dim=input_dim, out_dim=output_dim, n_hid=124)


def rnd_veto(input_file):
    """
    main func for AFL to call
    :param input_file:
    :return: true if RND has positive reward
    """
    if rnd_model is None:
        init_models()

    global step_counter
    step_counter += 1

    #if step_counter > retrain_every_x_seeds:
    #    update_model()
    #    step_counter = 0

    # convert file into byte-array
    byte_array = np.fromfile(input_file)
    # byte_array = vectorize

    if len(byte_array) > max_filesize:
        byte_array = byte_array[:max_filesize]
    else:
        byte_array = np.pad(byte_array, (0, max_filesize - len(byte_array)), 'constant',
                            constant_values=0)

    reward = rnd_model.get_reward(byte_array).detach().clamp(-1.0, 1.0).item()

    global replay_buffer
    replay_buffer.append(byte_array)

    step_counter += 1

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

