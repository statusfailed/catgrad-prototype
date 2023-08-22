import numpy as np
from yarrow import *
from yarrow.finite_function import cumsum
from yarrow.decompose.frobenius import frobenius_decomposition

from yarrow.numpy.layer import layer

def decompose_wiring(d: 'Diagram'):
    # Assume 'd' is a Frobenius decomposition.
    # Then return an Operations whose s_type and t_type values are
    # actually the wi/wo maps, respectively.
    Fun = d._Fun
    Array = Fun._Array
    s_type = IndexedCoproduct(
        sources = FiniteFunction(None, bincount(d.G.xi).table),
        values = d.G.wi)
    t_type = IndexedCoproduct(
        sources = FiniteFunction(None, bincount(d.G.xo).table),
        values = d.G.wo)
    return Operations(d.G.xn, s_type, t_type)

def acyclic_decompose(d: 'Diagram'):
    # Put in convenient form
    d = frobenius_decomposition(d)

    # layer the diagram
    layering, completed = layer(d)
    is_acyclic = np.all(completed)
    assert is_acyclic

    # extract operations
    ops = decompose_wiring(d)
    return d, ops, layering

# NOTE: this is a bit of a hack.
# Instead of storing s_type and t_type in Operations,
# we're actually storing *node indices*!
def to_python(d: Diagram, ops: Operations, layering, model_name='model') -> str:
    # TODO: fix yarrow's cumsum to give n+1 values!
    src_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    src_ptr[1:] = np.cumsum(ops.s_type.sources.table)

    tgt_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    tgt_ptr[1:] = np.cumsum(ops.t_type.sources.table)

    args = ", ".join(f"x_{i}" for i in d.s.table)
    s = f"def {model_name}({args}):\n"

    # bit of a hack :-)
    s += "    from numpy import fromiter as array\n"

    for op in layering.argsort().table:
        # (ι_{op} >> ops.s_type.values).table[0]
        x_s = ops.s_type.values.table[src_ptr[op]:src_ptr[op+1]]
        x_t = ops.t_type.values.table[tgt_ptr[op]:tgt_ptr[op+1]]

        # lhs = op(arg)
        lhs = ", ".join(f"x_{i}" for i in x_t) if len(x_t) > 0 else "_"
        arg = ", ".join(f"x_{i}" for i in x_s)
        op_value = d.G.xn(op)
        s += f"    {lhs} = {op_value}({arg}) # f_{op}\n"

    returns = ", ".join(f"x_{i}" for i in d.t.table)
    s += f"    return {returns}"
    return s

def optic_to_python(d: Diagram, model_name):
    d, ops, layering = acyclic_decompose(d)
    return to_python(d, ops, layering, model_name=model_name)
