from catgrad.optic import make_optic
import catgrad.signature as sig

def mse(B):
    fwd = sig.copy(B)
    rev = sig.sub(B)
    return make_optic(fwd, rev, B)
