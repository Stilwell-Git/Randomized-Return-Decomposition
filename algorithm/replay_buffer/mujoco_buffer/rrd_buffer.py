import copy
import numpy as np

class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': []
        }
        self.length = 0
        self.sum_rews = 0

    def store_transition(self, info):
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['obs'].append(copy.deepcopy(info['obs_next']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(np.float32(info['done'])))
        self.length += 1
        self.sum_rews += info['rews']

        if info['real_done']:
            for key in self.ep.keys():
                self.ep[key] = np.array(self.ep[key])

    def sample(self):
        idx = np.random.randint(self.length)
        info = {
            'obs': copy.deepcopy(self.ep['obs'][idx]),
            'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        return info

    def rrd_sample(self, sample_size, store_coef=False):
        idx = np.random.choice(self.length, sample_size, replace=(sample_size>self.length))
        info = {
            'rrd_obs': self.ep['obs'][idx],
            'rrd_obs_next': self.ep['obs'][idx+1],
            'rrd_acts': self.ep['acts'][idx],
            'rrd_rews': [self.sum_rews/self.length]
        }
        if store_coef:
            if (sample_size<=self.length) and (self.length>1):
                info['rrd_var_coef'] = [1.0-float(sample_size)/self.length]
            else:
                # We do not handle the case with (sample_size>self.length).
                info['rrd_var_coef'] = [1.0 if self.length>1 else 0.0]
        return info

class ReplayBuffer_RRD:
    def __init__(self, args):
        self.args = args
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size

        self.ep = []
        self.ram_idx = []
        self.length = 0
        self.head_idx = 0
        self.in_head = True

    def store_transition(self, info):
        if self.in_head:
            new_ep = Trajectory(info['obs'])
            self.ep.append(new_ep)
        self.ep[-1].store_transition(info)
        self.ram_idx.append(self.ep_counter)
        self.length += 1

        if self.length>self.buffer_size:
            del_len = self.ep[0].length
            self.ep.pop(0)
            self.head_idx += 1
            self.length -= del_len
            self.ram_idx = self.ram_idx[del_len:]

        self.step_counter += 1
        self.in_head = info['real_done']
        if info['real_done']:
            self.ep_counter += 1

    def sample_batch(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        if rrd_batch_size==-1: rrd_batch_size = self.args.rrd_batch_size
        if rrd_sample_size==-1: rrd_sample_size = self.args.rrd_sample_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[], rrd_obs=[], rrd_obs_next=[], rrd_acts=[], rrd_rews=[])
        if self.args.rrd_bias_correction:
            batch['rrd_var_coef'] = []

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        for i in range(rrd_batch_size//rrd_sample_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].rrd_sample(rrd_sample_size, store_coef=self.args.rrd_bias_correction)
            for key in info.keys():
                batch[key].append(info[key])

        return batch
