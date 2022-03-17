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

    def sample(self):
        idx = np.random.randint(self.length)
        info = {
            'obs': copy.deepcopy(self.ep['obs'][idx]),
            'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [self.sum_rews],
            'done': [copy.deepcopy(self.ep['done'][idx])]
        }
        return info

class ReplayBuffer_IRCR:
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

        self.heap_ep = []
        self.heap_ram_idx = []
        self.heap_length = 0

    def update_val_interval(self):
        vals = [ep.sum_rews for ep in self.ep+self.heap_ep]
        self.fifo_max_val = np.max(vals)
        self.fifo_min_val = np.min(vals)

    def update_heap(self, new_ep):
        if len(self.heap_ep)>=10:
            min_id = 0
            for id, ep in enumerate(self.heap_ep):
                if ep.sum_rews<self.heap_ep[min_id].sum_rews:
                    min_id = id
            if self.heap_ep[min_id].sum_rews>new_ep.sum_rews:
                return
            else:
                self.heap_length -= self.heap_ep[min_id].length
                self.heap_ep.pop(min_id)
        self.heap_ep.append(copy.deepcopy(new_ep))
        self.heap_length += new_ep.length
        self.heap_ram_idx = []
        for id, ep in enumerate(self.heap_ep):
            self.heap_ram_idx += [id]*ep.length

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
            self.update_heap(self.ep[-1])
            self.update_val_interval()

    def sample_batch(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            if i < batch_size//2:
                idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
                info = self.ep[idx].sample()
            else:
                idx = self.heap_ram_idx[np.random.randint(self.heap_length)]
                info = self.heap_ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        batch['rews'] = (np.array(batch['rews'])-self.fifo_min_val)/(self.fifo_max_val-self.fifo_min_val)

        return batch
