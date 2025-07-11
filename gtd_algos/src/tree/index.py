from jax import tree_map


def all_but_last(arg):
    return tree_map(lambda x: x[:-1], arg)


def last(arg):
    return tree_map(lambda x: x[-1], arg)
