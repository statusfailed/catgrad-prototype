import numpy as np
from yarrow import *
from yarrow.functor.functor import *
from yarrow.functor.optic import *

from catgrad.signature import xn0
import catgrad.signature as sig
import catgrad.layer as layer

# Pair of functors
#   F(·) A → ●_{i ∈ N} 'A_i
#   R(·) A → ●_{i ∈ N} A'_i
class Optic(FrobeniusOpticFunctor):
    """ Implement an optic functor by using yarrow's FrobeniusOpticFunctor helper """
    def map_fwd_objects(self, objects) -> IndexedCoproduct:
        return identity_object_map(objects)

    def map_rev_objects(self, objects) -> IndexedCoproduct:
        return identity_object_map(objects)

    def residuals(self, ops: Operations) -> IndexedCoproduct:
        # NOTE: this should associate a residual to each operation.
        # This means for each x ∈ ops.xn, we have a *list* of objects.
        # We therefore return a segmented list with sum(sources) = len(ops.xn).
        return IndexedCoproduct.from_list(None, [ op.residual() for op, _, _ in ops ])

    def map_fwd_operations(self, ops: Operations):
        # we need to return the tensoring of operations, with Bfwd/M uninterleaved.
        fwds = (op.fwd() for op, _, _  in ops)
        fwds, coarities = zip(*( (f, len(f.type[1])) for f in fwds), strict=True)
        return Diagram.tensor_list(fwds), FiniteFunction(None, coarities)

    def map_rev_operations(self, ops: Operations) -> Diagram:
        revs = (op.rev() for op, _, _ in ops)
        revs, arities = zip(*( (r, len(r.type[0])) for r in revs), strict=True)
        return Diagram.tensor_list(revs), FiniteFunction(None, arities)

def make_optic(fwd: Diagram, rev: Diagram, residual):
    """ Make an optic from:

    * a forward map ``f : A → B ● M``
    * a reverse map ``r : M ● B' → A'``
    * a residual ``M``.

    Giving a map of type

        make_optic(f) : A ● A' → B ● B'

    NOTE: this assumes A = A' for all A.
    """
    # TODO: bit of a hack, this is duplicated code from signature module, but we
    # rely on `optic` there, so we can't import it.
    xn0 = FiniteFunction.initial(None, dtype='object')
    
    # Work out types.
    # Note that we assume that Afwd = Arev so the optic / lens is "simple".
    A = fwd.type[0]

    N = len(rev.type[0]) - len(residual)
    B = FiniteFunction.inj0(N, len(residual)) >> rev.type[0]
    id_B = Diagram.identity(B, xn0)

    # Then make this diagram:
    #
    #      ┌───┐
    #      │   ├─────────────── B
    # A ───┤ f │  M
    #      │   ├───┐  ┌───┐
    #      └───┘   └──┤   │
    #                 │ r ├──── A'
    # B' ─────────────┤   │
    #                 └───┘
    lhs = fwd @ id_B
    rhs = id_B @ rev
    c = (fwd @ id_B) >> (id_B @ rev)

    # now adapt so that A' and B' are bent around and swapped.
    s = (FiniteFunction.inj0(len(A), len(B)) >> c.s) + (FiniteFunction.inj1(len(B), len(A)) >> c.t)
    t = (FiniteFunction.inj0(len(B), len(A)) >> c.t) + (FiniteFunction.inj1(len(A), len(B)) >> c.s)
    d = Diagram(s, t, c.G)

    xn = FiniteFunction.initial(len(fwd.G.xn)) >> fwd.G.xn
    lhs = Diagram.half_spider(FiniteFunction.cointerleave(len(A)), A + A, xn)
    rhs = Diagram.half_spider(FiniteFunction.cointerleave(len(B)), B + B, xn).dagger()

    return lhs >> d >> rhs

# The identity optic
def identity(A: FiniteFunction):
    """ The identity optic for type A """
    xn = FiniteFunction.initial(None, dtype='object')
    return Diagram.identity(A + A, xn)

def adapt_optic(optic: Diagram):
    # require even length source/targets.
    N_s = len(optic.s)
    N_t = len(optic.t)
    assert N_s % 2 == 0
    assert N_t % 2 == 0

    N_s = N_s // 2
    N_t = N_t // 2

    s = FiniteFunction.interleave(N_s) >> optic.s
    t = FiniteFunction.cointerleave(N_t) >> optic.t

    s_ = (FiniteFunction.inj0(N_s, N_s) >> s) + (FiniteFunction.inj1(N_t, N_t) >> t)
    t_ = (FiniteFunction.inj0(N_t, N_t) >> t) + (FiniteFunction.inj1(N_s, N_s) >> s)
    return Diagram(s_, t_, optic.G)
