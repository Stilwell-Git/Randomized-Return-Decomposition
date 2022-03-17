from .basis_alg.dqn import DQN
from .basis_alg.ddpg import DDPG
from .basis_alg.td3 import TD3
from .basis_alg.sac import SAC

basis_algorithm_collection = {
    'dqn': DQN,
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC
}

from .ircr import IRCR
from .rrd import RRD

advanced_algorithm_collection = {
    'ircr': IRCR,
    'rrd': RRD
}

algorithm_collection = {
    **basis_algorithm_collection,
    **advanced_algorithm_collection
}

def create_agent(args):
    return algorithm_collection[args.alg](args)
