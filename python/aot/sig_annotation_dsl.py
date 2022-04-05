from itertools import product
from typing import Sequence, Iterator

from ._types import NamedVariantsMap
from .abstract_values import TYPE_MAKERS, DummyCudaDevice, AbstractPtr, AbstractValue


class ConfigKeys:
    POINTER_TYPE = "*"
    IGNORE_SYMBOL = "_"
    UNIQUE_VARIANT = "^"


def _duplicate_independet_type_variants(attr_vars, named_vars):
    """
    Type variants defined in a global scope (visible to all kernels).
    This function generates per kernel version of named variants.

    "^" suffix means use a unique named variant.
    E.g:
        A = 1, 8
        Types:          x: 32, y:i32, z:i32
        Type Variants:  A^,    A,     A
    y and z attributes will have coupled values, first attribute will get it's own values.
    in total |product([1,8], [1,8])| kernels will be compiled.
    """
    var_counts = {}
    new_named_vars_scope = {}
    new_attr_vars = []
    for v in attr_vars:
        if v[-1] == ConfigKeys.UNIQUE_VARIANT:
            var_name = v[:-1]
            if var_name not in var_counts:
                var_counts[var_name] = 0
            var_counts[var_name] += 1
            vc = var_counts[var_name]
            # TODO: handle var_name not in scope error message
            new_var_name = f"{var_name}{vc}"
        else:
            new_var_name = v
            var_name = v
        new_named_vars_scope[new_var_name] = named_vars[var_name]
        new_attr_vars.append(new_var_name)

    return new_attr_vars, new_named_vars_scope


def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def sig_generator(
    pointers: Sequence[str],
    attributes: Sequence[str],
    attr_vars: Sequence[str],
    named_vars: NamedVariantsMap,
    constants_scope: NamedVariantsMap = None
) -> Iterator[Sequence[AbstractValue]]:

    abstract_pointers = [AbstractPtr(p, DummyCudaDevice(0)) for p in pointers]
    abstract_attr_makers = [TYPE_MAKERS[ty] for ty in attributes]

    dup_attr_vars, dup_named_vars = _duplicate_independet_type_variants(
        attr_vars, named_vars
    )

    for concrete_val_dict in dict_product(dup_named_vars):
        concrete_vals = map(concrete_val_dict.__getitem__, dup_attr_vars)
        abstract_vals = [f(v) for f, v in zip(abstract_attr_makers, concrete_vals)]
        yield abstract_pointers[:] + abstract_vals
