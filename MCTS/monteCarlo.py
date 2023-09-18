from node import PutNode
import copy, time
import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTree(object):
    def __init__(self, environment, observation, size_seq, nmodel = None, search_depth=None, rollout_length=-1, credit=1):
        self.sim_env = environment
        self.root = PutNode(None, 1.0)
        self.observation = observation

        self.known_size_seq = size_seq
        self.rollout_length = rollout_length
        self.credit = credit
        self.c = 1

        self.subrt = 0
        self.reached_depth = -1
        self.nmodel = nmodel
        if search_depth is not None:
            self.max_depth = min(search_depth, len(self.known_size_seq)-1)
        else:
            self.max_depth = len(self.known_size_seq)-1

        # if there is only one know box size, we can't do simulation
        assert len(self.known_size_seq) >= 2

    def select(self):
        cur_node = self.root
        cur_depth = 0
        obs = self.observation
        sim2_env = copy.deepcopy(self.sim_env)

        while True:
            # Terminated: back up
            if cur_node.is_terminated():
                # without future
                value = 0
                break
            # Not Expanded: expand node
            if not cur_node.is_expanded():
                #print('expand:',cur_depth)
                pointer = cur_depth
                cur_node.expand(nmodel= self.nmodel,
                                box_size_list=self.known_size_seq[pointer:],
                                rollout_length=self.rollout_length,
                                credit=self.credit,
                                observation=obs,
                                sim_env=sim2_env)
                value = cur_node.value
                break
            # reached max depth: back up
            if cur_depth == self.max_depth:
                value = cur_node.value
                break
            # not leaf node: use tree policy
            cur_action, next_node = cur_node.choose_best(self.c)
            # Simulate time: take the action
            action_idx = cur_action
            obs, reward, done, _ = sim2_env.step([action_idx])
            next_node.reward = reward
            if done:
                self.subrt += 1
                if not next_node.is_terminated():
                    next_node.terminate()
                cur_node = next_node
                value = 0
                break
            cur_node = next_node
            cur_depth += 1

        if cur_depth > self.reached_depth:
            self.reached_depth = cur_depth
        self.backup(cur_node, value)

    def backup(self, leaf_node, value, gamma=1):
        cur_node = leaf_node
        while True:
            value = cur_node.reward + gamma * value
            cur_node.update(value)
            if cur_node.prev_node is not None:
                cur_node = cur_node.prev_node
                continue
            break

    def play(self, zeta):
        actions_visits = [(a, nd.n) for (a, nd) in self.root.next_nodes.items()]
        actions, visits = zip(*actions_visits)
        values = softmax(1.0 / zeta * np.log(np.array(visits) + 1e-10))
        actions_values = dict(zip(actions, values))
        return actions_values

    def get_policy(self, sim_times, zeta=1):
        start = time.clock()
        for i in range(sim_times):
            # print('simulation',i+1)
            self.select()
        end = time.clock()
        p = self.play(zeta)
        print('cost time', end-start)
        print("terminated node:", self.subrt)
        print('reached depth:', self.reached_depth)
        return p

    def sample_action(self, policy):
        if self.max_depth == 0:
            def get_p(key):
                return self.root.next_nodes[key].p
            max_action = max(self.root.next_nodes, key=get_p)
            return max_action
        poss = [pos for _, pos in policy.items()]
        actions = [key for key in policy.keys()]
        action = np.random.choice(actions, p=poss)
        return action

    def succeed(self, put_action, new_box_size, observation):
        put_action = int(put_action)
        self.known_size_seq.pop(0)
        self.known_size_seq.append(new_box_size)
        new_node = self.root.next_nodes.get(put_action)
        assert new_node is not None
        new_node.p = 1.0
        new_node.prev_node = None
        print('reused simulation times:', new_node.n)
        print('children node number:', len(new_node.next_nodes))
        self.observation = observation
        self.root = new_node
        self.reached_depth = -1
        self.subrt = 0