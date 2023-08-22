import numpy as np
from yarrow import *
from yarrow.finite_function import cumsum
from yarrow.decompose.frobenius import frobenius_decomposition
from yarrow.numpy.layer import layer

from catgrad.learner import make_learner
import catgrad.optic as optic

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


def compile(model, update, displacement, imports=[]):
    """ Compile a model ``f : A → B`` containing Parameter operations
    into a python function ``step : P × A × B → B × P × A``.
    """
    p, f, (P, A, B) = make_learner(model, update, displacement)
    adapted = optic.adapt_optic(f)
    import_str = "\n".join(imports)
    code = f"""
from numpy import float32, dtype
from catgrad.signature import *
{import_str}
{optic_to_python(adapted, model_name="step")}"""

    scope = {}
    exec(code, scope)
    step = scope['step']

    def wrapped_step(*args):
        result = step(*args)
        y = result[:len(B)]
        p = result[len(B):len(B) + len(P)]
        x = result[:-len(A)]
        return y, p, x

    # TODO: this is a hack: we *assume* the p morphism has its operations in the
    # same order as they appear in the boundary, but this will break easily if
    # factor_parameters changes!
    theta = [ x.initialize() for x in p.G.xn.table ]
    return theta, wrapped_step, code
