# Deep Reinforcement Learning with Gradient Eligibility Traces

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

***(Published in the Reinforcement Learning Journal (RLC 2025))***

#### 🏆 Outstanding Paper Award on The Theory of Reinforcement Learning

## Overview
This repository contains the code to reproduce the experiments present in our paper titled [Deep Reinforcement Learning with Gradient Eligibility Traces](https://openreview.net/pdf?id=LZAafvwVMa) which was accepted at (RLC 2025) 


## Installation
create a virtual env
install requirements jax, mujoco,gymnasium,..etc (check the requirement file)

```
$ pip install -e .
```

## Usage
The [gtd_algos/exp_configs](gtd_algos/exp_configs) folder contains default configuration files with the hyperparameters configurations used to generate the results in our experiments.

Note on the config files: We have both exp_seed and env_seed, the exp_seed is used for randomization within the agent such as action selection, and env_seed is used to seed the environment. We think that having separate seeds for both the agent and the environment is a good practice to make sure the agent and the environment decoupled.

### Forward-View Algorithms:
We have two forward-view algorithms: PPO and Gradient PPO.
#### Gradient PPO
For Gradient PPO, you need to run:
```
python gtd_algos/src/experiments/gym_exps/gradient_ppo_main.py --config_file=gtd_algos/exp_configs/mujoco_gradient_ppo.yaml
```
The default values in the ```mujoco_gradient_ppo.yaml``` will run one seed for the Ant-v4 environments, and you can easily to change the seed or env_name value to try out different seeds and different environments. 

Different variations of gradient ppo updates can also be configured from the config file:
1. To get TDRC($\lambda$) update (the default): set ```gradient_correction``` to True and ```reg_coeff``` to a value greater than 0.0. The ```reg_coeff``` is the $\beta$ coeffient in the paper.
2. To get TDC($\lambda$) update: set ```gradient_correction``` to True and ```reg_coeff``` to 0.0.
3. To get GTD2($\lambda$) update: set ```gradient_correction``` to False.

Note that we have an additional config parameter ```is_correction``` when this is true. Setting it to True would use the equations in Appendix A for importance sampling. However, in our experiments we set it to False by defaults.

#### PPO
For PPO, you need to run:
```
python gtd_algos/src/experiments/gym_exps/ppo_main.py --config_file=gtd_algos/exp_configs/mujoco_ppo.yaml
```

### Backward-View Algorithms:
We have three backward-view algorithms: our QRC($\lambda$), Watkins' Q($\lambda$), and StreamQ (Elsayed et al., 2024).

#### QRC($\lambda$)
For QRC($\lambda$), you need to run:
```
python gtd_algos/src/experiments/gym_exps/qrc_main.py --config_file=gtd_algos/exp_configs/minatar_qrclambda.yaml
```
As before, to get the other variations of gradient updates, the config file can be edited as follows:

1. To get QRC($\lambda$) update (the default): set ```gradient_correction``` to True and ```reg_coeff``` to a value greater than 0.0. (1.0 by defaults)
2. To get QC($\lambda$) update: set ```gradient_correction``` to True and ```reg_coeff``` to 0.0.
3. To get GQ2($\lambda$) update: set ```gradient_correction``` to False.



#### Watkins' Q($\lambda$)
For Q($\lambda$), you need to run:
```
python gtd_algos/src/experiments/gym_exps/q_lambda_main.py --config_file=gtd_algos/exp_configs/minatar_qlambda.yaml
```

#### StreamQ
We include the PyTorch implementation provided by (Elsayed et al., 2024), and we include our implementation in Jax, which we verified to reproduce the same results but is also 2x faster than the PyTorch implementation. 
For Jax version:
```
python gtd_algos/src/experiments/gym_exps/streamq_main.py --config_file=gtd_algos/exp_configs/minatar_streamq_jax.yaml
```

For PyTorch version:
```
python gtd_algos/src/experiments/gym_exps/streamq_pytorch.py
```


## Citation
Please cite our work if you find it useful:

```latex
@inproceedings{elelimy2025deep,
title={Deep Reinforcement Learning with Gradient Eligibility Traces},
author={Esraa Elelimy and Brett Daley and Andrew Patterson and Marlos C. Machado and Adam White and Martha White},
booktitle={Reinforcement Learning Conference},
year={2025},
}
```
