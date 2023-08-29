from typing import Callable, Any
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
    if not is_acyclic:
        raise ValueError("Diagram is not acyclic")

    # extract operations
    ops = decompose_wiring(d)
    return d, ops, layering

# NOTE: this is a bit of a hack.
# Instead of storing s_type and t_type in Operations,
# we're actually storing *node indices*!
def diagram_to_python(f: Diagram, function_name='f') -> str:
    """ Transform an acyclic Diagram into python code """
    # Decompose an acyclic diagram (raising a ValueError if cycles exist)
    d, ops, layering = acyclic_decompose(f)

    # TODO: fix yarrow's cumsum to give n+1 values!
    src_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    src_ptr[1:] = np.cumsum(ops.s_type.sources.table)

    tgt_ptr = np.zeros(len(ops.xn)+1, dtype='int64')
    tgt_ptr[1:] = np.cumsum(ops.t_type.sources.table)

    args = ", ".join(f"x_{i}" for i in d.s.table)
    s = f"def {function_name}({args}):\n"

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


# Compile an (acyclic) Diagram into an executable python function.
def compile(f: Diagram, function_name='f', prefix=[]) -> Callable[Any, Any]:
    """ Compile an acyclic Diagram built using the Catgrad signature """
    python = diagram_to_python(f, function_name=function_name)
    prefix_str = "\n".join(prefix)
    # FIXME: import hacks to make FiniteFunction's repr() work.
    code = f"""
from numpy import float32, dtype
from catgrad.signature import *
from numpy import fromiter as array
{prefix_str}
{python}"""
    scope = {}
    exec(code, scope)
    return scope[function_name], code


def compile_model(model, update, displacement, prefix=[]):
    """ Compile a model ``f : A → B`` containing Parameter operations
    into a python function ``step : P × A × B → B × P × A``.
    """
    # fstar is the (unadapted) assembled model as an optic, including both
    # forward and reverse passes.
    # f is just the forward pass of the model, but note that it has parameters factored out!
    p, fstar, f, (P, A, B) = make_learner(model, update, displacement)
    adapted = optic.adapt_optic(fstar)
    step, code = compile(adapted, function_name='step', prefix=prefix)

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
    return theta, wrapped_step, f, code
