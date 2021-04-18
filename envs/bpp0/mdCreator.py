import numpy as np
import copy
import transforms3d.euler as eu
import random
import torch
from .binCreator import BoxCreator

class Box(object):
    def __init__(self, given_bound, high, low, x, y, z, lx, ly, lz):
        self.high_bound = high
        self.low_bound = low
        self.x = x
        self.y = y
        self.z = z
        self.volume = x * y * z
        self.location = np.array([lx, ly, lz])
        self.extent = np.array([x / 2, y / 2, z / 2])
        self.rotation = np.array([1, 0, 0, 0])
        self.Rot = eu.quat2euler(self.rotation)
        self.vertex = np.zeros((8, 3))
        self.getCorners(self.extent * 2, self.location)
        self.centre = (self.vertex[7] - self.vertex[0]) / 2 + self.location
        self.x_flag = False
        self.y_flag = False
        self.given_bound = given_bound

    def getCorners(self, size, location, quaternion=np.array([1, 0, 0, 0])):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.vertex[i * 4 + j * 2 + k] = np.array(
                        [location[0] + k * size[0], location[1] + j * size[1], location[2] + i * size[2]])
        R = eu.quat2mat(quaternion)
        vertex = np.array(self.vertex, np.float32)
        return np.dot(R, vertex.transpose())

    def rotate_box(self, quaternion):
        vertex = []
        if isinstance(quaternion, np.ndarray):
            if np.shape(quaternion) == (4, 1):
                self.rotation = quaternion.transpose()
            elif np.shape(quaternion) == (1, 4):
                self.rotation = quaternion
        elif isinstance(quaternion, list):
            self.rotation = np.array(quaternion)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertex.append([k * self.x, j * self.y, i * self.z])
                    # (0 0 0)(1 0 0)(0 1 0)(1 1 0)(0 0 1)(1 0 1)(0 1 1)(1 1 1)
        self.R = eu.quat2mat(quaternion)
        vertex = np.array(vertex, np.float32)
        vertex = np.dot(self.R, vertex.transpose()).transpose()
        for s in vertex:
            s += self.location
        self.vertex = vertex
        self.centre = (self.vertex[7] - self.vertex[0]) / 2 + self.location

    def benchmark_split(self):
        flags = []
        if self.x > self.given_bound[1]:
            flags.append(0)
        if self.y > self.given_bound[1]:
            flags.append(1)
        if self.z > self.given_bound[1]:
            flags.append(2)

        divide_flag = random.choice(flags)

        if divide_flag == 0:
            if self.x <= self.given_bound[0]:
                return False
            rand_x = random.randint(1, self.x)
            if rand_x < self.given_bound[0] or self.x - rand_x < self.given_bound[0]:
                return False
            box1 = Box(self.given_bound,self.high_bound, self.low_bound, rand_x, self.y, self.z, self.location[0], self.location[1],
                       self.location[2])
            box2 = Box(self.given_bound,self.high_bound, self.low_bound, self.x - rand_x, self.y, self.z, self.location[0] + rand_x,
                       self.location[1], self.location[2])
        elif divide_flag == 1:
            if self.y < self.given_bound[0]:
                return False
            rand_y = random.randint(1, self.y)
            if rand_y < self.given_bound[0] or self.y - rand_y < self.given_bound[0]:
                return False
            box1 = Box(self.given_bound,self.high_bound, self.low_bound, self.x, rand_y, self.z, self.location[0], self.location[1],
                       self.location[2])
            box2 = Box(self.given_bound,self.high_bound, self.low_bound, self.x, self.y - rand_y, self.z, self.location[0],
                       self.location[1] + rand_y, self.location[2])
        else:
            if self.z < self.given_bound[0]:
                return False
            rand_z = random.randint(1, self.z)
            if rand_z < self.given_bound[0] or self.z - rand_z < self.given_bound[0]:
                return False
            box1 = Box(self.given_bound,self.high_bound - rand_z, self.low_bound, self.x, self.y, self.z - rand_z, self.location[0],
                       self.location[1], self.low_bound)
            box2 = Box(self.given_bound,self.high_bound, self.high_bound - rand_z, self.x, self.y, rand_z, self.location[0],
                       self.location[1], self.high_bound - rand_z)
        return box1, box2


class bin(object):
    def __init__(self, container_size, given_bound):
        self.bin = Box(given_bound,container_size[2], 0, container_size[0], container_size[1], container_size[2],0,0,0)
        self.boxes = [self.bin]
        self.given_bound = given_bound
        print(self.given_bound)

    def is_valid(self, box):
        if box.x <= self.given_bound[1] and box.x >= self.given_bound[0] and \
                box.y <= self.given_bound[1] and box.y >= self.given_bound[0] \
                and box.z <= self.given_bound[1] and box.z >= self.given_bound[0]:
            return True
        return False

    def gen_benchmark(self):
        vaild_box = []
        invalid_box = copy.deepcopy(self.boxes)
        while True:
            for box in invalid_box:
                divide_boxes = box.benchmark_split()
                if not isinstance(divide_boxes, bool):
                    invalid_box.remove(box)
                    for sub_box in divide_boxes:
                        if self.is_valid(sub_box):
                            vaild_box.append(sub_box)
                        else:
                            invalid_box.append(sub_box)
            if len(invalid_box) == 0:
                self.boxes = vaild_box
                if len(vaild_box) <= 10:
                    print("???")
                break
        return True

    def depart_box(self):
        self.boxes.sort(key=lambda x: x.low_bound, reverse=False)

    def reset(self):
        while True:
            self.boxes = [self.bin]
            if self.gen_benchmark():
                break
        self.depart_box()

class MDlayerBoxCreator(BoxCreator):
    def __init__(self, container_size, given_bound):
        super().__init__()
        self.container = bin(container_size, given_bound)
        self.update_index = 0

    def reset(self):
        self.box_list.clear()
        self.index = 0
        self.container.reset()
        self.boxes = self.container.boxes
        self.default_box_set = []
        for box in self.boxes:
            self.default_box_set.append([box.x, box.y, box.z])
        self.default_box_set.append([10, 10, 10])
        self.box_set = self.default_box_set

    def generate_box_size(self, **kwargs):
        self.box_list.append(self.box_set[self.index])
        self.index += 1