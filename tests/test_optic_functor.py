from yarrow import *
from yarrow.functor.optic import *

import catgrad.signature as sig
from catgrad.optic import *

def test_optic_functor_singleton_diagram():
    T = sig.NdArray((5,4), int)
    X = sig.obj(T)

    c = sig.negate(X)
    d = Optic().map_arrow(c)
