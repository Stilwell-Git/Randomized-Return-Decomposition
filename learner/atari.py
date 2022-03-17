import copy
import numpy as np
from envs import make_env

class AtariLearner:
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.learner_info = [
            'Epsilon'
        ]
        self.ep = []

        args.eps_act = args.eps_l
        self.eps_decay = (args.eps_l-args.eps_r)/args.eps_decay

    def learn(self, args, env, agent, buffer):
        for _ in range(args.iterations):
            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                args.eps_act = max(args.eps_r, args.eps_act-self.eps_decay)
                obs, reward, done, _ = env.step(action)
                self.step_counter += 1
                frame = env.get_frame()
                transition = {
                    'obs': obs_pre,
                    'obs_next': obs,
                    'frame_next': frame,
                    'acts': action,
                    'rews': reward if args.env_type=='ep_rews' else np.clip(reward, -args.rews_scale, args.rews_scale),
                    'done': done
                }
                self.ep.append(transition)
                if done:
                    self.ep_counter += 1
                    for transition_ep in self.ep:
                        transition = copy.deepcopy(transition_ep)
                        buffer.store_transition(transition)
                    self.ep = []
                    obs = env.reset()
            args.logger.add_record('Epsilon', args.eps_act)

            if buffer.step_counter>=args.warmup:
                agent.normalizer_update(buffer.sample_batch())
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                    self.target_count += 1
                    if self.target_count%args.train_target==0:
                        agent.target_update()
