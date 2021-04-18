from numpy.random import randint
from copy import deepcopy

def load(file_path):
    f = open(file_path, 'r')
    all_trajs = []
    finish = False
    while True:
        f.readline()
        traj = []
        while True:
            box = f.readline()
            box = box.strip('\n').split(' ')
            if len(box) == 6:
                box = [int(x) for x in box]
                traj.append([box[0],box[1],box[2]])
            else:
                if len(traj) != 0:
                    all_trajs.append(traj)
                else:
                    finish = True
                break
        if finish == True:
            break
    return all_trajs

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            assert isinstance(self, RandomBoxCreator)
            self.generate_box_size()
        return deepcopy(self.box_list[:length])

    def get_box_size(self):
        assert len(self.box_list) >= 0
        next_box = self.box_list.pop(0)
        return next_box


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
        idx = randint(0, len(self.box_set)) # 随机抽一个
        self.box_list.append(self.box_set[idx])

class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name = None):
        super().__init__()
        self.box_trajs = load(data_name)
        self.box_index = 0
        self.traj_nums = len(self.box_trajs)
        self.index = randint(0,self.traj_nums)

    def reset(self):
        self.box_list.clear()
        self.index = randint(0,self.traj_nums)
        self.boxes = self.box_trajs[self.index]
        self.recorder = []
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        self.box_list.append(self.box_set[self.box_index])
        self.recorder.append(self.box_set[self.box_index])
        self.box_index += 1