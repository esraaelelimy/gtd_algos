import typing as t
from typing import Callable, Optional, Dict, Any
from flax import struct
from collections.abc import MutableMapping
import itertools

def flax_struct_to_dict(target):
    target_dict = {}
    for field_info in struct.dataclasses.fields(target):
        target_dict[field_info.name] = getattr(target, field_info.name)
    return target_dict
    

#this function is copied from https://gist.github.com/Microsheep/11edda9dee7c1ba0c099709eb7f8bea7
def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def rebuild(flatten_dict, result=None):
    from collections import defaultdict
    import json

    def tree():
        return defaultdict(tree)

    def rec(keys_iter, value):
        _r = tree()
        try:
            _k = next(keys_iter)
            _r[_k] = rec(keys_iter, value)
            return _r
        except StopIteration:
            return value

    if result is None:
        result = dict()

    for k, v in flatten_dict.items():
        keys_nested_iter = iter(k.split('__'))
        cur_level_dict = result
        while True:
            try:
                k = next(keys_nested_iter)
                if k in cur_level_dict:
                    cur_level_dict = cur_level_dict[k]
                else:
                    cur_level_dict[k] = json.loads(json.dumps(rec(keys_nested_iter, v)))
            except StopIteration:
                break

    return result


def get_configurations(params):
    # get all parameter configurations for individual runs
    list_params = [key for key in params.keys() if type(params[key]) is list]
    param_values = [params[key] for key in list_params]
    hyper_param_settings = list(itertools.product(*param_values))
    return list_params, hyper_param_settings
