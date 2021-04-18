import copy
import numpy as np
import random
import torch


class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def get_box_size(self):
        assert len(self.box_list) >= 0
        next_box = self.box_list.pop(0)
        return next_box

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)


class TestBoxCreator(BoxCreator):
    def __init__(self, data_url, shape):
        super().__init__()
        assert isinstance(data_url, str)
        data = np.loadtxt(data_url, dtype='int')
        data = np.reshape(data, newshape=shape)
        self.dataset = data
        self.data_pointer = 0
        self.mem_pointer = 0

    def generate_box_size(self, **kwargs):
        if self.mem_pointer < len(self.dataset[self.data_pointer]):
            next_box = tuple(self.dataset[self.data_pointer, self.mem_pointer])
            assert len(next_box) == 3
            self.box_list.append(next_box)
            self.mem_pointer += 1
        else:
            self.box_list.append((4, 4, 4))

    def reset(self):
        super().reset()
        self.mem_pointer = 0
        self.data_pointer = (self.data_pointer + 1) % len(self.dataset)


class RecallBoxCreator(BoxCreator):
    def __init__(self, memory_list):
        super().__init__()
        self.memory_list = memory_list
        self.memory_pointer = 0

    def generate_box_size(self, **kwargs):
        if self.memory_pointer < len(self.memory_list):
            self.box_list.append(self.memory_list[self.memory_pointer])
            self.memory_pointer += 1
        else:
            self.box_list.append((4, 4, 4))

    def reset(self):
        super().reset()
        self.memory_pointer = 0


class RandomBoxCreator(BoxCreator):
    default_box_set = \
        [
            (2, 2, 2), (2, 2, 3), (2, 2, 4),
            (2, 3, 2), (2, 3, 3), (2, 3, 4),
            (2, 4, 2), (2, 4, 3), (2, 4, 4),

            (3, 2, 2), (3, 2, 3), (3, 2, 4),
            (3, 3, 2), (3, 3, 3), (3, 3, 4),
            (3, 4, 2), (3, 4, 3), (3, 4, 4),

            (4, 2, 2), (4, 2, 3), (4, 2, 4),
            (4, 3, 2), (4, 3, 3), (4, 3, 4),
            (4, 4, 2), (4, 4, 3), (4, 4, 4),

        ]

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])


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
    def __init__(self, bin_size, box_range):
        super().__init__()
        self.box_list = []
        self.bin_size = bin_size
        self.box_range = box_range

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
        self.box_list.append((box.x, box.y, box.z))
        self._update(box)
        self._add_candidate()


class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1