from itertools import product
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Sequence, Tuple, Union
from triton.code_gen import JITFunction

from .abstract_values import (
    AbstractInt,
    AbstractFloat,
    AbstractBool,
    AbstractPtr,
    AbstractValue,
    DummyCudaDevice,
    TYPE_MAKERS,
)

from ._types import ModuleScope, NamedVariantsMap


class ConfigKeys:
    NAMED_VARIANTS = "named_type_variants"
    COMPILATION_CONFIG = "compile_params"
    KERNEL_SIGNITURES = "kernels"
    INPUT_TYPES = "types"
    TYPE_VARIANTS = "type_variants"
    META_INPUT = "meta"
    POINTER_TYPE = "*"
    UNIQUE_VARIANT = "^"
    IGNORE_SYMBOL = "_"


def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


AbstractValueMaker = Union[AbstractInt, AbstractFloat, AbstractBool]
MetaValue = Union[int, float, bool, str]
TypeVariant = str


@dataclass
class CompileParams:
    num_warps: int = 4
    num_stages: int = 4
    force_nc_cache: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def abstract_signature_generator(
    pointers: Sequence[AbstractPtr],
    variants: Sequence[TypeVariant],
    named_variants: NamedVariantsMap,
    abs_value_makers: Sequence[AbstractValueMaker],
) -> Iterator[Sequence[AbstractValue]]:
    for prod_vals in dict_product(named_variants):
        abstract_vals = [
            f(v) for f, v in zip(abs_value_makers, map(prod_vals.__getitem__, variants))
        ]
        yield pointers[:] + abstract_vals


@dataclass
class KernelSignitureConfig:
    name: str
    pointers: Sequence[AbstractPtr]
    abs_val_makers: Sequence[AbstractValueMaker]
    meta: Mapping[str, MetaValue]
    named_variants: NamedVariantsMap
    variants: Sequence[TypeVariant]
    compile_params: CompileParams

    def signiture_iter(self):
        """ Iterate over all variants of abstract inputs """
        return abstract_signature_generator(
            pointers=self.pointers,
            variants=self.variants,
            named_variants=self.named_variants,
            abs_value_makers=self.abs_val_makers,
        )


def parse_named_variants(global_named_variants: Mapping[str, str]) -> NamedVariantsMap:
    """
    Named variants are values assigned to kernel inputs for optimnization purposes.
    Triton optimizes kernels based on input data size. In AOT compilation data size is not known at compile time.
    Users can define several value options for input params and Triton will generate a kernel for each size combination.

    To allow coupling of values among inputs, we allow named definition of possible variants.

    Turn string variants into integers
    """
    # TODO: parse error messages
    return {
        var_name: [int(x) for x in global_named_variants[var_name].split(",")]
        for var_name in global_named_variants
    }


def parse_meta(meta_kwargs: Dict, module_scope: ModuleScope):
    """
    Parse meta attributes passed to kernels.
    We allow floats, ints and strings that represent in scope JITFunctions.

    """
    if module_scope is None:
        module_scope = {}
    meta = {}
    errors = {}
    for k, v in meta_kwargs.items():
        if isinstance(v, str):
            if isinstance(module_scope.get(v), JITFunction):
                meta[k] = module_scope.get(v)
                continue
            errors[k] = v
        elif isinstance(v, (float, int)):
            meta[k] = v
        else:
            errors[k] = v
    if len(errors):
        # TODO: proper error msgs
        msg = "\t"
        for k, v in errors.items():
            msg += f"{k} is {type(v)}"
            if isinstance(v, str):
                msg += f" ({v} not in scope as a JITFunction)"
            msg += "\n"
        raise ValueError(
            f"only numbers and JITFunctions are allowed as Meta args.\n Following keys are not compatible:\n {msg} "
        )

    return meta


def _parse_type_size_variants(
    var_ident: Sequence[str], nv: NamedVariantsMap
) -> Tuple[Sequence[str], NamedVariantsMap]:
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
    private_var_count = 0
    locl_nv = {}
    variant_keys = []
    for v in var_ident:
        if v == ConfigKeys.IGNORE_SYMBOL:
            continue
        if v[-1] == ConfigKeys.UNIQUE_VARIANT:
            # Create unique name for
            name = f"vv{private_var_count}"
            private_var_count += 1
            # TODO: propper named variant selection
            locl_nv[name] = nv[v[:-1]]
            variant_keys.append(name)
        elif v in nv:
            locl_nv[v] = nv[v]
            variant_keys.append(v)

    return variant_keys, locl_nv


def prep_data_for_signature_generators(
    tys: Sequence[str], ty_vars: Sequence[str], named_variants: NamedVariantsMap
):
    pointers = []
    type_makers = []

    for ty in tys:
        # Pointer type (e.g. Tensor)
        if ty[-1] == ConfigKeys.POINTER_TYPE:
            # TODO: device hard coded to 0, needs config
            dtype = AbstractPtr(ty[:-1], DummyCudaDevice(0))
            pointers.append(dtype)
        else:
            # create abstract Int/Float/Bool and leave the abstract value to be filled by type size variants
            ty_cls = TYPE_MAKERS[ty]
            type_makers.append(ty_cls)

    variant_keys, local_named_vars = _parse_type_size_variants(ty_vars, named_variants)

    return pointers, type_makers, variant_keys, local_named_vars


def parse_kernel(
    name: str,
    fconf: Mapping,
    named_variants: Mapping[str, Sequence[int]],
    compile_conf: CompileParams,
    module_scope: ModuleScope = None,
) -> KernelSignitureConfig:
    """
    Parse config files and prepare abstract kernel inputs for compilation optimization.
    """

    tys = [v.strip() for v in fconf.get(ConfigKeys.INPUT_TYPES, "").split(",")]
    ty_vars = [v.strip() for v in fconf.get(ConfigKeys.TYPE_VARIANTS, "").split(",")]

    (
        pointers,
        type_makers,
        variant_keys,
        local_named_vars,
    ) = prep_data_for_signature_generators(tys, ty_vars, named_variants)

    meta = parse_meta(fconf.get(ConfigKeys.META_INPUT, {}), module_scope)

    # TODO: have better explanation of what went wrong
    assert len(type_makers) == len(
        variant_keys
    ), f"In {name} function config got {len(type_makers)} values and {len(variant_keys)} variants. \n\t Values and Variants must be of same length"

    return KernelSignitureConfig(
        name=name,
        pointers=pointers,
        abs_val_makers=type_makers,
        variants=variant_keys,
        meta=meta,
        named_variants=local_named_vars,
        compile_params=compile_conf,
    )


def parse_compilation_config(
    conf: Dict, module_scope: ModuleScope = None
) -> Mapping[str, KernelSignitureConfig]:

    named_variants = parse_named_variants(conf.get(ConfigKeys.NAMED_VARIANTS, {}))
    compile_conf = CompileParams.from_dict(conf.get(ConfigKeys.COMPILATION_CONFIG, {}))

    return {
        func_name: parse_kernel(
            func_name,
            fconf,
            named_variants,
            compile_conf=compile_conf,
            module_scope=module_scope,
        )
        for func_name, fconf in conf.get(ConfigKeys.KERNEL_SIGNITURES, {}).items()
    }


def sig_generator(
    pointers: Sequence[str],
    attributes: Sequence[str],
    attr_vars: Sequence[str],
    named_vars: NamedVariantsMap,
) -> Iterator[Sequence[AbstractValue]]:

    sig = [f"{p}{ConfigKeys.POINTER_TYPE}" for p in pointers] + attributes
    type_variants = [ConfigKeys.IGNORE_SYMBOL] * len(pointers) + attr_vars

    (
        pointers,
        type_makers,
        variant_keys,
        local_named_vars,
    ) = prep_data_for_signature_generators(sig, type_variants, named_vars)

    return abstract_signature_generator(
        pointers=pointers,
        variants=variant_keys,
        abs_value_makers=type_makers,
        named_variants=local_named_vars,
    )

