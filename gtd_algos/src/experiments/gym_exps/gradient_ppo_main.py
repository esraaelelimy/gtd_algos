import wandb 

from gtd_algos.src.algorithms.gradient_ppo import GradientPPOAgent
from gtd_algos.src.experiments.main import main
from gtd_algos.src.experiments.gym_exps.ppo_main import experiment, define_metrics


if __name__ == "__main__":
    main(
        experiment, GradientPPOAgent, define_metrics,
        default_config_path='gtd_algos/exp_configs/mujoco_gradient_ppo.yaml',
    )