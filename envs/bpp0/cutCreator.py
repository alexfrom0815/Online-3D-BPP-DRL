import numpy as np
import copy
import random
import torch
from .binCreator import BoxCreator

class MetaBox():
    def __init__(self, x, y, z, lx, ly, lz):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def split(self, divide_flag, pos):
        if divide_flag == 0:
            box1 = MetaBox(pos, self.y, self.z, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x - pos, self.y, self.z, self.lx + pos, self.ly, self.lz)
        elif divide_flag == 1:
            box1 = MetaBox(self.x, pos, self.z, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x, self.y - pos, self.z, self.lx, self.ly + pos, self.lz)
        elif divide_flag == 2:
            box1 = MetaBox(self.x, self.y, pos, self.lx, self.ly, self.lz)
            box2 = MetaBox(self.x, self.y, self.z - pos, self.lx, self.ly, self.lz + pos)
        return box1, box2

    def __str__(self):
        return '(%d, %d, %d, %d, %d, %d)' % (self.x, self.y, self.z, self.lx, self.ly, self.lz)


class CuttingBoxCreator(BoxCreator):
    def __init__(self, bin_size, box_range, rotation=False):
        super().__init__()
        self.box_list = []
        self.bin_size = bin_size
        self.box_range = box_range
        self.rotation = rotation

        self.plain = np.zeros(shape=(self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self.meta_list = [MetaBox(*self.bin_size, 0, 0, 0)]
        self.candidates = []
        self._cut_box(*self.box_range)
        self._add_candidate()

    def reset(self):
        self.box_list.clear()
        self.plain = np.zeros(shape=(self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self.meta_list = [MetaBox(*self.bin_size, 0, 0, 0)]
        self.candidates = []
        self._cut_box(*self.box_range)
        self._add_candidate()

    def _check_box(self, box, low_x, low_y, low_z, high_x, high_y, high_z):
        x_flag = box.x < low_x or box.x > high_x
        y_flag = box.y < low_y or box.y > high_y
        z_flag = box.z < low_z or box.z > high_z
        return x_flag * 1 + y_flag * 2 + z_flag * 4

    def _choose_pos(self, box, check, low_x, low_y, low_z, high_x, high_y, high_z):
        df_list = []
        if 1 & check:
            df_list.append(0)
        if 2 & check:
            df_list.append(1)
        if 4 & check:
            df_list.append(2)
        df = random.choice(df_list)
        if df == 0:
            pos_range = (low_x, box.x - low_x)
        if df == 1:
            pos_range = (low_y, box.y - low_y)
        if df == 2:
            pos_range = (low_z, box.z - low_z)
        assert pos_range[0] <= pos_range[1]
        pos = random.randint(pos_range[0], pos_range[1])
        return df, pos

    def _cut_box(self, low_x, low_y, low_z, high_x, high_y, high_z):
        continue_flag = True
        new_list = []
        while continue_flag:
            continue_flag = False
            for box in self.meta_list:
                check = self._check_box(box, low_x, low_y, low_z, high_x, high_y, high_z)
                if check == 0:
                    new_list.append(box)
                else:
                    df, pos = self._choose_pos(box, check, low_x, low_y, low_z, high_x, high_y, high_z)
                    box1, box2 = box.split(df, pos)
                    new_list.append(box1)
                    new_list.append(box2)
                    continue_flag = True
            self.meta_list = copy.deepcopy(new_list)
            new_list.clear()
            # print('total box num: ', len(self.meta_list))

    def _add_candidate(self):
        new_list = []
        for i in range(len(self.meta_list)):
            mb = self.meta_list[i]
            check = (self.plain[mb.lx:mb.lx + mb.x, mb.ly:mb.ly + mb.y] == mb.lz).sum() - mb.x * mb.y
            if check == 0:
                self.candidates.append(mb)
            else:
                new_list.append(mb)
        self.meta_list = new_list

    def _update(self, box):
        self.plain[box.lx:box.lx + box.x, box.ly:box.ly + box.y] += box.z

    def generate_box_size(self, **kwargs):
        if len(self.candidates) == 0:
            self.box_list.append(self.bin_size)
            return
        idx = random.randint(0, len(self.candidates) - 1)
        box = self.candidates.pop(idx)
        if not self.rotation:
            self.box_list.append((box.x, box.y, box.z))
        else:
            rd = np.random.rand()
            # randomly rotate boxes
            if rd < 0.5:
                self.box_list.append((box.x, box.y, box.z))
            else:
                self.box_list.append((box.y, box.x, box.z))
        self._update(box)
        self._add_candidate()

class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name = None):
        super().__init__()
        self.box_trajs = torch.load(data_name)
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(self.box_trajs)

    def reset(self):
        self.box_list.clear()
        self.boxes = self.box_trajs[self.index]
        self.recorder = []
        self.index += 1
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        self.box_list.append(self.box_set[self.box_index])
        self.recorder.append(self.box_set[self.box_index])
        self.box_index += 1

