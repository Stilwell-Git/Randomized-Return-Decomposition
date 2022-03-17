import copy
import numpy as np

class Episode_FrameStack:
    def __init__(self, info):
        self.common_info = [
            'obs', 'obs_next', 'frame_next',
            'acts', 'rews', 'done'
        ]
        self.ep = {
            'obs': [],
            'acts': [],
            'rews': [],
            'done': []
        }
        for key in info.keys():
            if not(key in self.common_info):
                self.ep[key] = []
        self.ep_len = 0
        self.sum_rews = 0.0
        self.frames = info['obs'].shape[-1]
        for i in range(self.frames):
            self.ep['obs'].append(copy.deepcopy(info['obs'][:,:,i]))

    def insert(self, info):
        self.ep_len += 1
        self.sum_rews += info['rews']
        self.ep['obs'].append(copy.deepcopy(info['frame_next']))
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(info['done']))
        for key in info.keys():
            if not(key in self.common_info):
                self.ep[key].append(copy.deepcopy(info[key]))

    def get_obs(self, idx):
        idx += 1
        obs = np.stack(self.ep['obs'][idx:idx+self.frames], axis=-1)
        return obs.astype(np.float32)/255.0

    def sample(self):
        idx = np.random.randint(self.ep_len)
        info = {
            'obs': self.get_obs(idx-1),
            'obs_next': self.get_obs(idx),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        for key in self.ep.keys():
            if (not(key in self.common_info)) and (not(key in info.keys())):
                info[key] = copy.deepcopy(self.ep[key][idx])
        return info

    def sample_ircr(self):
        idx = np.random.randint(self.ep_len)
        info = {
            'obs': self.get_obs(idx-1),
            'obs_next': self.get_obs(idx),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [self.sum_rews],  # critical step of IRCR
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        for key in self.ep.keys():
            if (not(key in self.common_info)) and (not(key in info.keys())):
                info[key] = copy.deepcopy(self.ep[key][idx])
        return info

    def sample_rrd(self, sample_size, store_coef=False):
        idx = np.random.choice(self.ep_len, sample_size, replace=(sample_size>self.ep_len))
        info = {
            'rrd_obs': [],
            'rrd_obs_next': [],
            'rrd_acts': [],
            'rrd_rews': [self.sum_rews/self.ep_len]
        }
        for _ in range(sample_size):
            idx = np.random.randint(self.ep_len)
            info['rrd_obs'].append(self.get_obs(idx-1))
            info['rrd_obs_next'].append(self.get_obs(idx))
            info['rrd_acts'].append(copy.deepcopy(self.ep['acts'][idx]))
        if store_coef:
            if (sample_size<=self.ep_len) and (self.ep_len>1):
                info['rrd_var_coef'] = [1.0-float(sample_size)/self.ep_len]
            else:
                info['rrd_var_coef'] = [1.0 if self.ep_len>1 else 0.0]
        return info

class ReplayBuffer_FrameStack:
    def __init__(self, args):
        self.args = args
        self.in_head = True
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size

        self.ep = []
        self.length = 0
        self.head_idx = 0
        self.ram_idx = []

        self.sample_batch = {
            'dqn': self.sample_batch_dqn,
            'ircr': self.sample_batch_ircr,
            'rrd': self.sample_batch_rrd,
        }[args.alg]

    def store_transition(self, info):
        if self.in_head:
            new_ep = Episode_FrameStack(info)
            self.ep.append(new_ep)
        self.ep[-1].insert(info)
        self.ram_idx.append(self.ep_counter)
        self.length += 1

        if self.length>self.buffer_size:
            del_len = self.ep[0].ep_len
            self.ep.pop(0)
            self.head_idx += 1
            self.length -= del_len
            self.ram_idx = self.ram_idx[del_len:]

        self.step_counter += 1
        self.in_head = info['done']
        if info['done']:
            self.ep_counter += 1

    def sample_batch_dqn(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        return batch

    def sample_batch_ircr(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample_ircr()  # critical step of IRCR
            for key in info.keys():
                batch[key].append(info[key])

        return batch

    def sample_batch_rrd(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
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
            info = self.ep[idx].sample_rrd(rrd_sample_size, store_coef=self.args.rrd_bias_correction)
            for key in info.keys():
                batch[key].append(info[key])

        return batch
