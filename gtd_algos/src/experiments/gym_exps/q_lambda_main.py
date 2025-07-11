from gtd_algos.src.algorithms.q_lambda import QLambdaAgent
from gtd_algos.src.experiments.gym_exps.streamq_main import define_metrics, experiment
from gtd_algos.src.experiments.main import main


if __name__ == "__main__":
    main(
        experiment, QLambdaAgent, define_metrics,
        default_config_path='gtd_algos/exp_configs/minatar_qlambda.yaml',
    )
