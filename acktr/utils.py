import glob
import os
import torch.nn as nn
import numpy as np
from acktr.envs import VecNormalize


def check_box(plain, x, y, lx, ly, z, container_size):
    if lx + x > container_size[0] or ly + y > container_size[1]:
        return -1
    if lx < 0 or ly < 0:
        return -1

    rec = plain[lx:lx + x, ly:ly + y]
    max_h = np.max(rec)
    max_area = np.sum(rec == max_h)
    area = x * y

    assert max_h >= 0
    if max_h + z > container_size[2]:
        return -1

    LU = int(rec[0, 0] == max_h)
    LD = int(rec[x - 1, 0] == max_h)
    RU = int(rec[0, y - 1] == max_h)
    RD = int(rec[x - 1, y - 1] == max_h)

    if max_area / area > 0.95:
        return max_h
    if LU + LD + RU + RD == 3 and max_area / area > 0.85:
        return max_h
    if LU + LD + RU + RD == 4 and max_area / area > 0.50:
        return max_h

    return -1

def get_possible_position(observation, container_size):
    if not isinstance(observation, np.ndarray):
        box_info = observation.cpu().numpy()
    else:
        box_info = observation
    box_info = box_info.reshape((4,-1))
    x = int(box_info[1][0])
    y = int(box_info[2][0])
    z = int(box_info[3][0])

    plain = box_info[0].reshape((container_size[0],container_size[1]))

    width = container_size[0]
    length = container_size[1]

    action_mask = np.zeros(shape=(width, length), dtype=np.int32)

    for i in range(width - x + 1):
        for j in range(length - y + 1):
            if check_box(plain, x, y, i, j, z, container_size) >= 0:
                action_mask[i, j] = 1

    if action_mask.sum() == 0:
        action_mask[:, :] = 1

    return action_mask.reshape((-1,)).tolist()

def get_rotation_mask(observation, container_size):
    box_info = observation.cpu().numpy()
    box_info = box_info.reshape((4,-1))
    x = int(box_info[1][0])
    y = int(box_info[2][0])
    z = int(box_info[3][0])

    plain = box_info[0].reshape((container_size[0],container_size[1]))

    width = container_size[0]
    length = container_size[1]

    action_mask1 = np.zeros(shape=(width, length), dtype=np.int32)
    action_mask2 = np.zeros(shape=(width, length), dtype=np.int32)

    for i in range(width - x + 1):
        for j in range(length - y + 1):
            if check_box(plain, x, y, i, j, z, container_size) >= 0:
                action_mask1[i, j] = 1

    for i in range(width - y + 1):
        for j in range(length - x + 1):
            if check_box(plain, y, x, i, j, z, container_size) >= 0:
                action_mask2[i, j] = 1

    action_mask = np.hstack((action_mask1.reshape((-1,)), action_mask2.reshape((-1,))))

    if action_mask.sum() == 0:
        action_mask[:] = 1

    return action_mask

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
