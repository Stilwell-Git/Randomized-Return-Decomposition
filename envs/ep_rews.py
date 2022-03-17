import numpy as np

def create_EpisodicRewardsEnv(basis_env):
    class EpisodicRewardsEnv(basis_env):
        def __init__(self, args):
            super().__init__(args)

        def step(self, action):
            obs, reward, done, info = self.env_step(action)
            self.rews_buffer.append(reward)
            if done:
                reward = np.sum(self.rews_buffer)
                self.rews_buffer = []
            else:
                reward = 0.0
            return obs, reward, done, info

        def reset(self):
            self.rews_buffer = []
            return super().reset()

    return EpisodicRewardsEnv
