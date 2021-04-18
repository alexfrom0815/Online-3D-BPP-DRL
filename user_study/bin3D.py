from space import Space
from binCreator import LoadBoxCreator
import numpy as np
from copy import deepcopy
from space import Box

class PackingGame(object):
    def __init__(self, box_creator=None, enable_give_up=False):
        self.box_creator = box_creator
        self.space = Space(10, 10, 10)
        self.can_give_up = enable_give_up
        self.next_box = None
        self.cur_observation = None
        self.data_name = 'cut_2.txt'
        if self.box_creator is None:
            self.box_creator = LoadBoxCreator(data_name = self.data_name)

    def reset(self):
        pass

    def step(self, action):
        idx = action[0]
        ratio = self.space.get_ratio()
        if idx >= self.space.get_action_space():
            if self.can_give_up:
                observation = self.cur_observation
                reward = ratio
                done = True
                info = dict()
                info['counter'] = len(self.space.boxes)
                info['ratio'] = ratio
                return observation, reward, done, info
            else:
                raise Exception('out of the boundary of action space')
        succeeded = self.space.drop_box(self.next_box, idx)

        if not succeeded:
            observation = self.cur_observation
            reward = -1.0
            done = True
            info = dict()
            info['counter'] = len(self.space.boxes)
            info['ratio'] = ratio
            return observation, reward, done, info

        ratio = self.space.get_ratio()
        self.box_creator.get_box_size()
        self.box_creator.generate_box_size()
        self.next_box = self.box_creator.preview(1)[0]

        observation = np.array([*self.space.plain.reshape
        (shape=(self.space.get_action_space(),)),*self.next_box, ratio])

        self.cur_observation = observation

        reward = 0.0
        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['ratio'] = ratio
        return observation, reward, done, info

    def get_possible_position(self):
        pass


class AdjustPackingGame(PackingGame):
    def __init__(self, box_creator=None, enable_give_up=False, adjust_grid=0, **kwags):
        super().__init__(box_creator, enable_give_up)
        self.adjust_grid = adjust_grid
        if kwags.get('_adjust_ratio'):
            self.adjust_grid = kwags.get('_adjust_ratio')
        self.flip_possibility = kwags.get('flip_possibility')

        self.adjust_flag = kwags.get('adjust')
        self.container = Box(10,10,10,0,0,0)
        self.container.set_color('skyblue')

        if self.flip_possibility is not None:
            self.UD_flip = False
            self.LR_flip = False

    def reset(self):
        self.box_creator.reset()
        self.space = Space(10, 10, 10)
        self.box_creator.generate_box_size()
        self.next_box = self.box_creator.preview(1)[0]
        self.temp_box = deepcopy(self.next_box)
        self.cur_observation = np.array(
            [*np.reshape(self.space.plain, newshape=(self.space.get_action_space(),)),
             *self.next_box, self.space.get_ratio()])
        return np.append(self.cur_observation, self.get_possible_position(self.adjust_flag))

    def get_possible_position(self, plain=None):
        x = self.next_box[0]
        y = self.next_box[1]
        z = self.next_box[2]

        if plain is None:
            plain = self.space.plain

        width = self.space.plain_size[0]
        length = self.space.plain_size[1]

        action_mask = np.zeros(shape=(width, length), dtype=np.int32)

        for i in range(width):
            for j in range(length):
                if self.space.check_box(plain, x, y, i, j, z) >= 0:
                    action_mask[i, j] = 1

        if action_mask.sum() == 0:
            action_mask[:, :] = 1

        return action_mask

    def get_flip(self):
        if np.random.random() < self.flip_possibility:
            print('flip_UD')
            self.UD_flip = True
        if np.random.random() < self.flip_possibility:
            print('flip_LR')
            self.LR_flip = True

    def augment_observation(self, plain):
        assert self.flip_possibility is not None
        if self.UD_flip and self.LR_flip:
            return np.flip(plain)
        if self.UD_flip and not self.LR_flip:
            return np.flipud(plain)
        if not self.UD_flip and self.LR_flip:
            return np.fliplr(plain)
        else:
            return plain

    def transfer_action(self, action):
        if not self.UD_flip and not self.LR_flip:
            return action
        bin_size = self.space.plain_size
        box_size = self.next_box
        lx, ly = self.space.idx_to_position(action)
        if self.UD_flip and not self.LR_flip:
            lx = bin_size[0] - lx - box_size[0]
        if self.LR_flip and not self.UD_flip:
            ly = bin_size[1] - ly - box_size[1]
        if self.UD_flip and self.LR_flip:
            lx = bin_size[0] - lx - box_size[0]
            ly = bin_size[1] - ly - box_size[1]
        transfered_action = self.space.position_to_index((lx, ly))
        self.UD_flip = False
        self.LR_flip = False
        return transfered_action

    def try_step(self, action):
        idx = action[0]
        succeeded = self.space.try_drop(self.temp_box, idx)
        return succeeded

    def step(self, action):
        idx = action[0]
        if self.flip_possibility is not None:
            idx = self.transfer_action(idx)

        if idx >= self.space.get_action_space():
            if self.can_give_up:
                ratio = self.space.get_ratio()
                observation = np.append(self.cur_observation, self.get_possible_position(self.adjust_flag))
                reward = ratio * 10  # todo
                done = True
                info = dict()
                info['mask'] = self.get_possible_position(self.adjust_flag)
                info['counter'] = len(self.space.boxes)
                info['ratio'] = ratio
                return observation, reward, done, info
            else:
                raise Exception('out of the boundary of action space')

        succeeded = self.space.drop_box(self.next_box, idx)

        if not succeeded:
            ratio = self.space.get_ratio()
            observation = np.append(self.cur_observation, self.get_possible_position(self.adjust_flag))
            if self.can_give_up:
                reward = 0.0
            else:
                reward = ratio * 10
            done = True
            info = dict()
            info['counter'] = len(self.space.boxes)
            info['ratio'] = ratio
            info['mask'] = self.get_possible_position(self.adjust_flag)
            return observation, reward, done, info

        self.box_creator.get_box_size()
        self.box_creator.generate_box_size()
        self.next_box = self.box_creator.preview(1)[0]
        self.temp_box = deepcopy(self.next_box)

        plain = self.space.plain
        if self.flip_possibility is not None:
            self.get_flip()
            plain = self.augment_observation(plain)

        observation = np.array([*np.reshape(plain, newshape=(-1,)),
                                *self.next_box, self.space.get_ratio()])
        self.cur_observation = observation

        mask = self.get_possible_position(self.adjust_flag)
        observation = np.append(self.cur_observation, mask)
        reward = 0
        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['ratio'] = self.space.get_ratio()
        # info['dis'] = dis
        info['mask'] = mask
        return observation, reward, done, info

    def _get_dis(self, mov):
        return int(np.linalg.norm(mov, ord=1))

    def _min_mov(self, point, targets, lx, ly):
        min_dis = 1000
        min_vec = None
        for target in targets:
            target = np.array(target, dtype=np.int32)
            cur_vec = target - point
            cur_dis = self._get_dis(cur_vec)
            if cur_dis <= self.adjust_grid and cur_dis < min_dis:
                plain = self.space.plain
                x = self.next_box[0]
                y = self.next_box[1]
                z = self.next_box[2]
                adj_lx = lx + cur_vec[0]
                adj_ly = ly + cur_vec[1]
                if self.space.check_box(plain, x, y, adj_lx, adj_ly, z) >= 0:
                    min_dis = cur_dis
                    min_vec = cur_vec
        return min_vec, min_dis

    def adjust(self, idx):
        movec = np.zeros(shape=2, dtype=np.int32)
        dis = 1000

        lx, ly = self.space.idx_to_position(idx)
        x = self.next_box[0]
        y = self.next_box[1]

        guad = self.space.get_corners()
        lu = np.array([lx, ly], dtype=np.int32)
        ld = np.array([lx + x, ly], dtype=np.int32)
        ru = np.array([lx, ly + y], dtype=np.int32)
        rd = np.array([lx + x, ly + y], dtype=np.int32)
        mov_lu, dis_lu = self._min_mov(lu, guad[3], lx, ly)
        mov_ld, dis_ld = self._min_mov(ld, guad[0], lx, ly)
        mov_ru, dis_ru = self._min_mov(ru, guad[2], lx, ly)
        mov_rd, dis_rd = self._min_mov(rd, guad[1], lx, ly)

        def func(cur_mov, cur_dis, mov, min_dis):
            if cur_dis < min_dis:
                mov = cur_mov
                min_dis = cur_dis
            return mov, min_dis

        movec, dis = func(mov_lu, dis_lu, movec, dis)
        movec, dis = func(mov_ld, dis_ld, movec, dis)
        movec, dis = func(mov_ru, dis_ru, movec, dis)
        movec, dis = func(mov_rd, dis_rd, movec, dis)
        position = (lx + movec[0], ly + movec[1])
        return self.space.position_to_index(position), dis


