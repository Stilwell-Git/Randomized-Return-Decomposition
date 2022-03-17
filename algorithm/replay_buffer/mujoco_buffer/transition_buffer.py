import copy
import numpy as np

class ReplayBuffer_Transition:
    def __init__(self, args, buffer_size=None):
        self.args = args
        self.buffer = {}
        self.length = 0
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size if buffer_size is None else buffer_size

    def store_transition(self, info):
        if self.step_counter==0:
            for key in info.keys():
                self.buffer[key] = []
        if self.step_counter<self.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key].append(copy.deepcopy(info[key]))
            self.length += 1
        else:
            idx = self.step_counter%self.buffer_size
            for key in self.buffer.keys():
                self.buffer[key][idx] = copy.deepcopy(info[key])
        self.step_counter += 1
        self.ep_counter += 1 if info['real_done'] else 0

    def sample_batch(self, batch_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            idx = np.random.randint(self.length)
            for key in self.buffer.keys():
                if key in ['rews', 'done']:
                    batch[key].append(copy.deepcopy([np.float32(self.buffer[key][idx])]))
                elif key in ['obs', 'obs_next', 'acts']:
                    batch[key].append(copy.deepcopy(self.buffer[key][idx]))

        return batch
