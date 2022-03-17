# Randomized Return Decomposition (RRD)

This is a TensorFlow implementation for our paper [Learning Long-Term Reward Redistribution via Randomized Return Decomposition](https://arxiv.org/abs/2111.13485) accepted by ICLR 2022.

## Requirements
1. Python 3.6.13
2. gym == 0.18.3
3. TensorFlow == 1.12.0
4. BeautifulTable == 0.8.0
5. opencv-python == 4.5.3.56

## Running Commands

Run the following commands to reproduce our main results shown in section 4.1.

```bash
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
python train.py --tag='RRD-L(RD) Ant-v2' --alg=rrd --basis_alg=sac --rrd_bias_correction=True --env=Ant-v2
```

The following commands to switch the back-end algorithm of RRD.

```bash
python train.py --tag='RRD-TD3 Ant-v2' --alg=rrd --basis_alg=td3 --env=Ant-v2
python train.py --tag='RRD-DDPG Ant-v2' --alg=rrd --basis_alg=ddpg --env=Ant-v2
```

We include an *unofficial* implementation of IRCR for the ease of baseline comparison.  
Please refer to [tgangwani/GuidanceRewards](https://github.com/tgangwani/GuidanceRewards) for the official implementation of IRCR.

```bash
python train.py --tag='IRCR-SAC Ant-v2' --alg=ircr --basis_alg=sac --env=Ant-v2
python train.py --tag='IRCR-TD3 Ant-v2' --alg=ircr --basis_alg=td3 --env=Ant-v2
python train.py --tag='IRCR-DDPG Ant-v2' --alg=ircr --basis_alg=ddpg --env=Ant-v2
```

The following commands support the experiments on Atari games with episodic rewards.  

```bash
python train.py --tag='RRD-DQN Assault' --alg=rrd --basis_alg=dqn --env=Assault
python train.py --tag='IRCR-DQN Assault' --alg=ircr --basis_alg=dqn --env=Assault
```

**Note:**
The implementation of RRD upon DQN on the Atari benchmark has not been well tuned. We release this interface only for the ease of future studies.
