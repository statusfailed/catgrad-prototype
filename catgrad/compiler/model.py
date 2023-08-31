from typing import Callable, Any
from yarrow import Diagram

from catgrad.compiler.decompose import acyclic_decompose_operations
from catgrad.compiler.python import to_module, to_function
from catgrad.learner import make_learner
from catgrad import optic

def compile_diagram(f: Diagram, function_name='f', module_name='catgrad.dynamic') -> Callable[Any, Any]:
    ops = list(acyclic_decompose_operations(f))
    decorators = []
    tree = to_module(to_function(function_name, f.s.table, f.t.table, ops, decorators))

    code = compile(tree, module_name, mode='exec')
    scope = {}
    exec(code, scope)
    return scope[function_name]

def compile_model(model, update, displacement, prefix=[]):
    """ Compile a model ``f : A → B`` containing Parameter operations
    into a python function ``step : P × A × B → B × P × A``.
    """
    # fstar is the (unadapted) assembled model as an optic, including both
    # forward and reverse passes.
    # f is just the forward pass of the model, but note that it has parameters factored out!
    p, fstar, f, (P, A, B) = make_learner(model, update, displacement)
    adapted = optic.adapt_optic(fstar)

    assert prefix == []
    step = compile_diagram(adapted, function_name='step')

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
    return theta, wrapped_step, f
