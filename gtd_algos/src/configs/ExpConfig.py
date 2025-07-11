from .utils import transform_dict, flax_struct_to_dict
from flax import struct
import typing as t
import argparse
import yaml

from gtd_algos.src.configs.Config import Config


@struct.dataclass
class ExpConfig:
    tag: str = struct.field(pytree_node=False) # a tag to identify the experiment in wandb
    wandb_project_name: str = struct.field(pytree_node=False)
    exp_seed: int = struct.field(pytree_node=False)
    algo: str = struct.field(pytree_node=False)
    agent_config: Config = struct.field(pytree_node=False)
    env_config: Config = struct.field(pytree_node=False)

    @classmethod
    def from_dict(cls: t.Type["ExpConfig"], obj: dict):
        return cls(
            tag=obj["tag"],
            wandb_project_name=obj["wandb_project_name"],
            exp_seed=obj["exp_seed"],
            algo=obj["algo"],
            agent_config=Config.from_dict(obj["agent_config"]),
            env_config=Config.from_dict(obj["env_config"]),
        )

    def to_dict(self, expand: bool = True):
        exp_dict = flax_struct_to_dict(self)
        exp_dict["agent_config"] = flax_struct_to_dict(self.agent_config)
        exp_dict["env_config"] = flax_struct_to_dict(self.env_config)
        return transform_dict(exp_dict, expand)

    @staticmethod
    def from_yaml(config_file: str = 'configs/gymnax_config.yaml'):
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)    
        return ExpConfig.from_dict(config)
