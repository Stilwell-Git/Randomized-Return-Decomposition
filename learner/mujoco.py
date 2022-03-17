import copy
import numpy as np
from envs import make_env

class MuJoCoLearner:
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.learner_info = []
        self.ep = []

    def learn(self, args, env, agent, buffer):
        for _ in range(args.iterations):
            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                obs, reward, done, info = env.step(action)
                transition = {
                    'obs': obs_pre,
                    'obs_next': obs,
                    'acts': action,
                    'rews': reward,
                    'done': done if env.steps<args.test_timesteps else False,
                    'real_done': done
                }
                self.step_counter += 1
                self.ep.append(transition)
                if done:
                    self.ep_counter += 1
                    for transition_ep in self.ep:
                        transition = copy.deepcopy(transition_ep)
                        buffer.store_transition(transition)
                    self.ep = []
                    obs = env.reset()

            if buffer.step_counter>=args.warmup:
                agent.normalizer_update(buffer.sample_batch())
                for _ in range(args.train_batches):
                    self.target_count += 1
                    if 'pi_delay_freq' in args.__dict__.keys():
                        batch = buffer.sample_batch()
                        info = agent.train_q(batch)
                        args.logger.add_dict(info)
                        if self.target_count%args.pi_delay_freq==0:
                            batch = buffer.sample_batch()
                            info = agent.train_pi(batch)
                            args.logger.add_dict(info)
                    else:
                        batch = buffer.sample_batch()
                        info = agent.train(batch)
                        args.logger.add_dict(info)
                    if self.target_count%args.train_target==0:
                        agent.target_update()
