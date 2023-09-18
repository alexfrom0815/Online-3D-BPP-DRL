import numpy as np
import math, copy, time


INF = 1e9+7

class Node:
    def __init__(self, prev, p):
        self.prev_node = prev
        self.next_nodes = {}
        self.terminated = False
        self.value = None
        self.reward = 0

        self.q = 0
        self.w = 0
        self.n = 0
        self.p = p

    def is_expanded(self):
        return len(self.next_nodes) > 0

    def is_terminated(self):
        return self.terminated

    def terminate(self):
        self.terminated = True
        self.p = 0

    def update(self, value):
        self.n += 1
        self.w += value
        # normal average
        self.q = self.w / self.n
        # moving average

    def get_u_value(self):
        u_value = self.p * np.sqrt(self.prev_node.n)/(self.n+1)
        return u_value

    def get_q_value(self):
        return self.q

    # def check_terminate(self):
    #     ps = []
    #     for _, node in self.next_nodes.items():
    #         ps.append(node.p)
    #     ps = np.array(ps)
    #     p_sum = np.sum(ps)
    #     if math.isclose(p_sum, 1.0, rel_tol=1e-3):
    #         return False
    #     ps = ps / p_sum
    #     ps = ps.tolist()
    #     for _, node in self.next_nodes.items():
    #         p_new = ps.pop(0)
    #         p_old = node.p
    #         node.p = p_new
    #         assert p_old >= p_new
    #     return True

    def choose_best(self, c=1):
        assert len(self.next_nodes) > 0
        max_value = -INF
        max_nodes = []

        for (action, node) in self.next_nodes.items():
            if node.n > 0:
                advanced_q = node.get_q_value() - node.prev_node.get_q_value()
                cur_value = advanced_q + c * node.get_u_value()
            else:
                cur_value = 0.0 + c * node.get_u_value()
            if math.isclose(cur_value, max_value, rel_tol=1e-5):
                max_nodes.append((action, node))
            elif cur_value > max_value:
                max_value = cur_value
                max_nodes.clear()
                max_nodes.append((action, node))
        assert len(max_nodes) > 0
        idx = np.random.randint(0, len(max_nodes))
        return max_nodes[idx]

    def expand(self, **kwargs):
        pass


class PutNode(Node):

    def __init__(self, prev, p):
        super().__init__(prev, p)
        self.q = 0

    def expand(self, nmodel, **kwargs):

        credit = kwargs.get('credit')
        rollout_length = kwargs.get('rollout_length')
        box_size_list = kwargs.get('box_size_list')
        observation = kwargs.get('observation')
        sim_env = kwargs.get('sim_env')
        
        assert box_size_list is not None
        assert len(box_size_list) >= 1
        assert observation is not None
        assert sim_env is not None

        if credit is not None:
            assert credit <=1 and credit >=0
        else:
            credit = 1

        if rollout_length is not None:
            if rollout_length == -1:
                rollout_length = len(box_size_list)-1
        else:
            rollout_length = 0
        
        # get valid position
        action_mask = sim_env.get_possible_position()
        action_mask = np.reshape(action_mask, newshape=(-1,))
        
        # get possibilities using neural network
        value, pvec = nmodel.evaluate(observation, False)

        valid_action_num = np.sum(action_mask)
        
        for i in range(len(action_mask)):
            action = i
            if action_mask[i] == 1: # !!! still use mask !!!
                action_possibility = credit * pvec[action] + (1-credit) * (1/valid_action_num)
                self.next_nodes[action] = PutNode(self, action_possibility)

        # no give-up action, default action is '0'
        if len(self.next_nodes) == 0:
            self.next_nodes[0] = PutNode(self, 1)

        if rollout_length >= 1 and len(box_size_list) >= rollout_length + 1:
            value = self.roll_out(box_size_list[:rollout_length+1], copy.deepcopy(sim_env), observation, nmodel)
        self.value = value

    def roll_out(self, box_size_list, sim_env, observation, nmodel, gamma=1):
        assert box_size_list is not None
        assert sim_env is not None
        assert observation is not None

        # 做出的动作数量
        sim3_env = sim_env
        obs = observation
        box_num = len(box_size_list)
        reward_stack = []

        value = None
        for i in range(box_num):
            # box_size = box_size_list[i]
            value, prev = nmodel.evaluate(obs, False)

            # action_pos = dict(zip(range(prev.shape[0]), prev))
            # action_max = max(action_pos, key=action_pos.get)
            # obs, reward, done, _ = sim3_env.step([action_max])

            action_sample = np.random.choice(prev.shape[0], p=prev)
            obs, reward, done, _ = sim3_env.step([action_sample])

            if not done and i+1 < box_num:
                reward_stack.append(reward)
            if done:
                reward_stack.append(reward)
                value = 0
                break

        for i in range(len(reward_stack)-1, -1, -1):
            value = reward_stack[i] + gamma * value
        return value




