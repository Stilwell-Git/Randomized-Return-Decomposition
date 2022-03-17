from .atari import AtariLearner
from .mujoco import MuJoCoLearner

def create_learner(args):
    return {
        'atari': AtariLearner,
        'mujoco': MuJoCoLearner
    }[args.env_category](args)
