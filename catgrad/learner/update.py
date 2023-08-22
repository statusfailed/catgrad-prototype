import numpy as np
from yarrow import *
from catgrad import NdArray, layer, signature as sig
from catgrad.optic import make_optic

# gradient descent with a learning rate of 0.01.
def gd(ε=0.01, dtype='f4'):
    def gd_inner(P: FiniteFunction):
        fwd = sig.copy(P)

        # learning rates -- constants
        lr = np.array([ε], dtype=dtype)
        c = Diagram.tensor_list([sig.Constant(lr).to_diagram()] * len(P))
        C = c.type[1]

        id_P = sig.identity(P)

        mul = sig.multiply(P, C)
        scale = (id_P @ c) >> mul

        # NOTE: sub because we're computing a *distance*, not a point.
        rev = (id_P @ scale) >> sig.sub(P)

        return make_optic(fwd, rev, residual=fwd.type[0])
    return gd_inner
