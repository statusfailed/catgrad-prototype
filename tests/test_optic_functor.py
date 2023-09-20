from yarrow import *
from yarrow.functor.optic import *

import catgrad.signature as sig
from catgrad.optic import *

def test_optic_functor_singleton_diagram():
    T = sig.NdArray((5,4), int)
    X = sig.obj(T)

    c = sig.negate(X)
    d = Optic().map_arrow(c)

# The n > 3 case used to break the Optic functor, so we explicitly test it
def test_add3():
    A0 = sig.NdArray((1,2,3), int)
    A1 = sig.NdArray((4,5), int)
    A2 = sig.NdArray((6,), int)

    X = sig.obj(A0, A1, A2)
    d = sig.add(X)

    o = adapt_optic(Optic().map_arrow(d))

    assert o.type[0] == (X + X) + X
    assert o.type[1] == X + (X + X)
