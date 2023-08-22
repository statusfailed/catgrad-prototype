""" Given a circuit ``c : A → B`` with ``Parameter`` operations, replace them
with dangling wires to obtain a circuit ``unparametrise(c) : P × A → B``
"""
import numpy as np
from yarrow import *

from catgrad.signature import Parameter

# Given a diagram
#   f : A → B
# containing N `Parameter` operations,
# factor it into the composite
#
#   (p × id) ; f'
#
# where
#
#   p  : I → Σ_{n ∈ N} τ₁(Pn)
#   f' : Σ_{n ∈ N} τ₁(Pn) ● A → B
#
# where f' does not contain any Parameter operations.
def factor_parameters(c: Diagram):
    """ Transform a diagram ``c : A → B`` into one ``unparametrise(c) : P × A → B`` """
    # get Parameter operations
    param_ops = None

    # Which ops have type Parameter?
    parameter_ids = np.fromiter([type(x) is Parameter for x in c.G.xn.table], dtype='bool')

    if len(parameter_ids) == 0:
        p = Diagram.empty()
    else:
        p = Diagram.tensor_list([ p.to_diagram() for p in c.G.xn.table[parameter_ids] ])

    not_parameter_ids = FiniteFunction(2, 1 - parameter_ids)
    parameter_ids = FiniteFunction(2, parameter_ids + 0)

    # Which wires are in the *image* of parameter_ids?
    # NOTE: we can be a bit lazy here, because Parameter has (co)arity 0 → 1, we
    # can just treat this as a list of individual wires instead of "bundles"
    # (i.e., if we had operations of type 0 > N, for N > 1.
    # TODO: duplicate computation of d.G.xo >> p!
    op_targets = filter_by(c.G.wo, c.G.xo >> parameter_ids)

    # Remove Parameter ops from the diagram, but not the wires.
    d = filter_operations(c, not_parameter_ids)
    f = Diagram(op_targets + d.s, d.t, d.G)
    return p, f

# TODO: better name?
def filter_by(f: FiniteFunction, p: FiniteFunction):
    # f : X → Y
    # p : X → 2
    # filter(f, p) : Σ_{x ∈ X} p(x) → Y
    assert p.target == 2
    assert f.source == p.source
    return FiniteFunction(f.target, f.table[p.table.astype(bool)])

# p : X → Bool
def filter_operations(d: Diagram, p: FiniteFunction):
    """ Given a diagram ``d`` with ``X`` operations,
        and a predicate ``p : X → 2`, remove those operations ``x`` from ``d``
        where ``p(x) == 0``.
    """
    assert d.G.X == p.source
    # This is mostly just a pointwise 'filter_by' on maps.
    # However, for xi and xo, we have to use cumsum(p).
    # let's say we have
    #   xi = [ 0 0 1 0 2 2 ]
    # then filtering by the predicate
    #   p = [ 1 0 1 ]
    # will get us
    #   filter_by(xi,p) = [ 0 0 0 2 2 ]
    # but we need to renumber this - we should have 2 operations left.
    # Using cumsum:
    #   cumsum(p) = [ 0 1 1 ]
    # then we just post-compose:
    #   filter_by(xi,p) >> cumsum(p) = [ 0 0 0 1 1 ]
    pc = cumsum(p)
    p1 = d.G.xi >> p
    p2 = d.G.xo >> p

    xn = filter_by(d.G.xn, p)
    X = xn.source
    xi = FiniteFunction(X, (filter_by(d.G.xi, p1) >> pc).table)
    xo = FiniteFunction(X, (filter_by(d.G.xo, p2) >> pc).table)

    G = BipartiteMultigraph(
        wn = d.G.wn,
        wi = filter_by(d.G.wi, p1),
        xi = xi,
        pi = filter_by(d.G.pi, p1),

        wo = filter_by(d.G.wo, p2),
        xo = xo,
        po = filter_by(d.G.po, p2),

        xn = filter_by(d.G.xn, p))

    return Diagram(d.s, d.t, G)
