from flax import struct

from gtd_algos.src.configs.utils import transform_dict, flax_struct_to_dict


@struct.dataclass
class Config:
    d: dict = struct.field(pytree_node=False)

    def __getattribute__(self, name):
        d = object.__getattribute__(self, 'd')
        try:
            return d[name]
        except KeyError:
            return object.__getattribute__(self, name)

    @classmethod
    def from_dict(cls: "Config", d: dict):
        return cls(d)

    def to_dict(self, expand: bool = True):
        state_dict = flax_struct_to_dict(self)
        return transform_dict(state_dict, expand)
