import numpy as np
from functools import reduce
from copy import deepcopy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Box(object):
    def __init__(self, x, y, z, lx, ly, lz):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.color = None
        self.vertex = np.zeros((8, 3))
        self.refresh()

    def set_color(self,color):
        if self.color is None:
            self.color = color

    def refresh(self):
        self.getCorners([self.x, self.y, self.z], [self.lx, self.ly, self.lz])

    def getCorners(self, size, location, quaternion=np.array([1, 0, 0, 0])):  # 找八个定点的坐标
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.vertex[i * 4 + j * 2 + k] = np.array(
                        [location[0] + k * size[0], location[1] + j * size[1], location[2] + i * size[2]])
        vertex = np.array(self.vertex, np.float32)
        return vertex.transpose()

    def plot_opaque_cube(self, ax, text, color='red', alpha = 1):
        if self.color is not None:
            color = self.color

        self.refresh()
        ax.text3D(self.vertex[0][0], self.vertex[0][1], self.vertex[0][2], text, fontsize=15,
                  verticalalignment="center",
                  horizontalalignment="center")

        xx = np.array([self.lx, self.x + self.lx])
        yy = np.array([self.ly, self.y + self.ly])
        zz = np.array([self.lz, self.z + self.lz])

        xx, yy = np.meshgrid(xx, yy)

        ax.plot_surface(xx, yy, xx * 0 + self.lz, color=color, alpha=0.5)
        ax.plot_surface(xx, yy, xx * 0 + self.lz + self.z, color=color, alpha=0.5)

        yy, zz = np.meshgrid(yy, zz)
        ax.plot_surface(yy * 0 + self.lx, yy, zz, color=color, alpha=0.5)
        ax.plot_surface(yy * 0 + self.lx + self.x, yy, zz, color=color, alpha=0.5)

        xx, zz = np.meshgrid(xx, zz)
        ax.plot_surface(xx, zz * 0 + self.ly, zz, color=color, alpha=0.5)
        ax.plot_surface(xx, zz * 0 + self.ly + self.y, zz, color=color, alpha=0.5)



    def plot_linear_cube(self, ax, text, color='red', alpha = 1,linestyle = '-'):
        if self.color is not None:
            color = self.color
        self.refresh()
        ax.text3D(self.vertex[0][0], self.vertex[0][1], self.vertex[0][2], text, fontsize=15,
                  verticalalignment="center",
                  horizontalalignment="center")
        xd = [self.vertex[0][0], self.vertex[2][0], self.vertex[3][0], self.vertex[1][0], self.vertex[0][0]]
        yd = [self.vertex[0][1], self.vertex[2][1], self.vertex[3][1], self.vertex[1][1], self.vertex[0][1]]
        zd = [self.vertex[0][2], self.vertex[2][2], self.vertex[3][2], self.vertex[1][2], self.vertex[0][2]]
        xu = [self.vertex[4][0], self.vertex[6][0], self.vertex[7][0], self.vertex[5][0], self.vertex[4][0]]
        yu = [self.vertex[4][1], self.vertex[6][1], self.vertex[7][1], self.vertex[5][1], self.vertex[4][1]]
        zu = [self.vertex[4][2], self.vertex[6][2], self.vertex[7][2], self.vertex[5][2], self.vertex[4][2]]
        kwargs = {'alpha': 1, 'color': color,'linestyle':linestyle}
        ax.plot3D(xd, yd, zd, **kwargs)  # 绘制下表面
        ax.plot3D(xu, yu, zu, **kwargs)
        for i in range(4):
            ax.plot3D([self.vertex[i][0], self.vertex[i + 4][0]], [self.vertex[i][1], self.vertex[i + 4][1]],
                      [self.vertex[i][2], self.vertex[i + 4][2]], **kwargs)

    def standardize(self):
        return tuple([self.x, self.y, self.z, self.lx, self.ly, self.lz])


class Space(object):
    def __init__(self, width=10, length=10, height=10):
        self.plain_size = np.array([width, length, height])
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.boxes = []
        self.height = height
        self.try_box = None


    def print_height_graph(self):
        print(self.plain)

    def get_height_graph(self):
        plain = np.zeros(shape=self.plain_size[:2], dtype=np.int32)
        for box in self.boxes:
            plain = self.update_height_graph(plain, box)
        return plain

    @staticmethod
    def update_height_graph(plain, box):
        plain = deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.lz + box.z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    def get_plain(self):
        return deepcopy(self.plain)

    def get_action_space(self):
        return self.plain_size[0] * self.plain_size[1]

    def get_corners(self):
        width = self.plain_size[0]
        length = self.plain_size[1]
        guad = [list() for _ in range(4)]

        guad[0].append((width, 0))
        guad[1].append((width, length))
        guad[2].append((0, length))
        guad[3].append((0, 0))

        for i in range(1, width):
            if self.plain[i, 0] != self.plain[i - 1, 0]:
                guad[0].append((i, 0))
                guad[3].append((i, 0))

        for i in range(1, width):
            if self.plain[i, length - 1] != self.plain[i - 1, length - 1]:
                guad[1].append((i, length))
                guad[2].append((i, length))

        for j in range(1, length):
            if self.plain[0, j] != self.plain[0, j - 1]:
                guad[2].append((0, j))
                guad[3].append((0, j))

        for j in range(1, length):
            if self.plain[width - 1, j] != self.plain[width - 1, j]:
                guad[0].append((width, j))
                guad[1].append((width, j))

        for i in range(1, width):
            for j in range(1, length):
                grid_0 = self.plain[i - 1, j]
                grid_1 = self.plain[i - 1, j - 1]
                grid_2 = self.plain[i, j - 1]
                grid_3 = self.plain[i, j]
                if grid_0 == grid_1 and grid_2 == grid_3:
                    continue
                if grid_0 == grid_3 and grid_1 == grid_2:
                    continue
                if grid_0 != grid_3 or grid_0 != grid_1:
                    guad[0].append((i, j))
                if grid_1 != grid_0 or grid_1 != grid_2:
                    guad[1].append((i, j))
                if grid_2 != grid_1 or grid_2 != grid_3:
                    guad[2].append((i, j))
                if grid_3 != grid_2 or grid_3 != grid_0:
                    guad[3].append((i, j))

        return guad

    def check_box(self, plain, x, y, lx, ly, z):
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return -1
        if lx < 0 or ly < 0:
            return -1

        rec = plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)

        # check boundary
        assert max_h >= 0
        if max_h + z > self.height:
            return -1

        # check area and corner
        max_area = np.sum(rec == max_h)
        area = x * y

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

    def get_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def idx_to_position(self, idx):
        lx = idx // self.plain_size[1]
        ly = idx % self.plain_size[1]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.plain_size[0] and position[1] < self.plain_size[1]
        return position[0] * self.plain_size[1] + position[1]

    def try_drop(self, box_size, idx):
        self.try_box = None
        lx, ly = self.idx_to_position(idx)
        x = box_size[0]
        y = box_size[1]
        z = box_size[2]
        plain = self.plain
        new_h = self.check_box(plain, x, y, lx, ly, z)
        if new_h != -1:
            self.try_box = Box(x, y, z, lx, ly, new_h)
            return True
        return False

    def drop_box(self, box_size, idx):
        self.try_box = None
        lx, ly = self.idx_to_position(idx)
        x = box_size[0]
        y = box_size[1]
        z = box_size[2]
        plain = self.plain
        new_h = self.check_box(plain, x, y, lx, ly, z)
        if new_h != -1: # 摆放成功的话
            self.boxes.append(Box(x, y, z, lx, ly, new_h))
            self.plain = self.update_height_graph(plain, self.boxes[-1])
            self.height = max(self.height, new_h + z)
            return True
        return False

def plot_plate(ax, color='blue',size = [10,10,10]):
    xd = [0, 0, size[0], size[0], 0]
    yd = [0, size[0], size[0], 0, 0]
    zd = [0, 0, 0, 0, 0]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xd, yd, zd, **kwargs)

def plot(boxes, size = [10,10,10]):
    # import re
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_plate(ax)
    for i in range(len(boxes)):
        boxes[i].plot_linear_cube(ax, str(i))

    ax.set_xlim3d(0,size[0]*1.5)
    ax.set_ylim3d(0,size[0]*1.5)
    ax.set_zlim3d(0,size[0]*1.5)
    plt.show()
    plt.close()

