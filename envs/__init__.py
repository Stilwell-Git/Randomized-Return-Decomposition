from .normal_atari import AtariNormalEnv
from .normal_mujoco import MuJoCoNormalEnv
from .ep_rews import create_EpisodicRewardsEnv

atari_list = [
    'Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
    'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
    'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk',
    'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
    'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
    'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
    'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
    'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
    'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]

mujoco_list = [
    'Ant-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Humanoid-v2',
    'Reacher-v2', 'Swimmer-v2', 'Hopper-v2', 'HumanoidStandup-v2'
]

envs_collection = {
    # Atari envs
    **{
        atari_name : 'atari'
        for atari_name in atari_list
    },
    # MuJoCo envs
    **{
        mujoco_name : 'mujoco'
        for mujoco_name in mujoco_list
    }
}

def make_env(args):
    normal_env = {
        'atari': AtariNormalEnv,
        'mujoco': MuJoCoNormalEnv
    }[envs_collection[args.env]]

    return {
        'normal': normal_env,
        'ep_rews': create_EpisodicRewardsEnv(normal_env)
    }[args.env_type](args)
