from yarrow import *
from catgrad import *
import catgrad.signature as sig
from catgrad.signature import obj

from catgrad.optic import make_optic

def test_make_optic_no_residual():
    T = NdArray((3,), 'u8')
    U = T + 1
    fwd = sig.Reshape(T, U).to_diagram()
    rev = sig.Reshape(U, T).to_diagram()

    o = make_optic(fwd, rev, obj())

    # z = fwd.type[0] + rev.type[1]
    assert o.type[0] == FiniteFunction.cointerleave(1) >> (fwd.type[0] + rev.type[1])
    assert o.type[1] == FiniteFunction.cointerleave(1) >> (fwd.type[1] + rev.type[0])

def test_make_optic_copy_add():
    T = NdArray((5,4,3), int)
    fwd = sig.Copy(T).to_diagram()
    rev = sig.Add(T).to_diagram()

    o = make_optic(fwd, rev, obj(T))

    assert o.type[0] == fwd.type[0] + rev.type[1]
    assert o.type[1] == obj(T, T)

def test_make_optic_copy_add_interleaved():
    T = NdArray((5,4,3), int)
    U = NdArray((8,9), int)

    X = obj(T, U)
    fwd = sig.copy(X)
    rev = sig.add(X)

    o = make_optic(fwd, rev, X)

    assert o.type[0] == obj(T, T) + obj(U, U)
    assert o.type[1] == obj(T, T) + obj(U, U)

def test_lens_fwd():
    T = NdArray((5,4,3), int)
    U = NdArray((8,9), int)
    A = FiniteFunction(None, [T, U], 'O')
    c = sig.copy(A)

    d = sig.lens_fwd(c)

    assert d.type[0] == A
    assert d.type[1] == c.type[1] + A
