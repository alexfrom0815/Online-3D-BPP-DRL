import numpy as np
import copy

# FOR ENV 'MASK'
class ReorderTree(object):
    def __init__(self, nmodel, box_list, env, c_bound=0.1):
        # the box number used for reordering
        self.box_num = len(box_list)
        # the shape of the action mask
        self.mask_shape = env.bin_size[:2]
        self.mask_len = self.mask_shape[0] * self.mask_shape[1]
        # copy the env and box list
        self.env = copy.deepcopy(env)
        self.box_list = copy.deepcopy(box_list)
        # threshold
        self.c_bound = c_bound
        self.v_bound = 0.1
        self.nmodel = nmodel

    def get_mixed_obs(self, masks, real_idx, raw_obs):
        max_height = self.env.bin_size[-1]
        new_obs = copy.deepcopy(raw_obs).reshape(self.env.bin_size[:2])
        new_obs = new_obs.reshape(-1)
        for i in range(real_idx+1, self.box_num):
            new_obs = max_height * (1 - masks[i]) + new_obs * masks[i]
        return new_obs

    def update_mask(self, mask, box, pos):
        pos = (pos//self.mask_shape[1], pos%self.mask_shape[1])
        cmask = copy.deepcopy(mask).reshape(self.mask_shape)
        cmask[pos[0]:pos[0]+box[0], pos[1]:pos[1]+box[1]] = 0
        return cmask.reshape(-1)

    def evaluate(self, obs, masks, real_idx):
        # 4 channels
        revised_obs = copy.deepcopy(obs).reshape(4,-1)

        raw_obs = copy.deepcopy(revised_obs[0])
        new_obs = self.get_mixed_obs(masks, real_idx, raw_obs)

        revised_obs[0] = new_obs
        revised_obs = revised_obs.reshape((-1,))
        
        val, poss = self.nmodel.evaluate(revised_obs)
        action = np.argsort(poss)[-1]
        return val, action

    def search(self, masks, cur_env, res_idxs):
        max_exp = -10000
        max_act = None
        for idx in res_idxs:
            if idx == self.box_num-1 and len(res_idxs) > 1:
                continue
            cur_box = self.box_list[idx]
            tmp_env = copy.deepcopy(cur_env)
            tmp_env.box_creator.box_list = [cur_box, self.env.bin_size]
            tmp_obs = tmp_env.cur_observation
            val, pos = self.evaluate(tmp_obs, masks, idx)
            # prune branches
            # if val < max_exp - self.c_bound:
            #     continue
            # copy and update environment
            next_env = copy.deepcopy(tmp_env)
            _, reward, done, _ = next_env.step([pos])
            # current box can't be put
            if done:
                fail_flag = False
                for i in range(self.box_num):
                    if (i in res_idxs and i < idx) or (i not in res_idxs and i >= idx):
                        fail_flag = True
                        break
                if fail_flag:
                    continue
                if max_exp < 0:
                    max_exp = 0
                    if idx == 0:
                        max_act = 0
                    else:
                        max_act = None
                continue
            # reach the evaluation point
            if len(res_idxs) == 1:
                assert idx == self.box_num-1
                if max_exp < val:
                    max_exp = val
                    if idx == 0:
                        max_act = pos
                    else:
                        max_act = None
                continue
            # copy and update [res_idxs]
            next_idxs = copy.deepcopy(res_idxs)
            next_idxs.remove(idx)
            assert not done
            # copy and update [mask]
            next_masks = copy.deepcopy(masks)
            next_masks[idx] = self.update_mask(next_masks[idx], cur_box, pos)
            # recursion
            value, action = self.search(next_masks, next_env, next_idxs)
            if reward + value > max_exp:
                max_exp = reward + value
                if idx == 0:
                    max_act = pos
                elif action is not None:
                    assert 0 in res_idxs
                    max_act = action
        return max_exp, max_act

    def get_baseline(self):
        env = copy.deepcopy(self.env)
        obs = env.cur_observation
        nor_exp = 0
        nor_act = None
        area = self.mask_len
        for i in range(self.box_num):
            val, poss = self.nmodel.evaluate(obs)
            act = np.argmax(poss)
            if i == 0:
                nor_act = act
            obs, reward, done, info = env.step([act])
            if done:
                nor_exp += 0
                return nor_exp, nor_act
            if i == self.box_num-1:
                nor_exp += val
                return nor_exp, nor_act
            nor_exp += reward
        
    def reorder_search(self):
        nor_exp, nor_act = self.get_baseline()
        res_idxs = list(range(self.box_num))
        masks = np.ones((self.box_num, self.mask_len))
        max_exp, max_act = self.search(masks, self.env, res_idxs)
        if max_act != nor_act and max_exp - nor_exp < 0.1:
            max_exp = nor_exp
            max_act = nor_act
        default = (max_act==nor_act)
        return max_act, max_exp, default
