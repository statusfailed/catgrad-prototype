from catgrad.signature import NdArray, obj
from catgrad.learner.update import gd
from catgrad.learner.displacement import mse

def test_gd():
    P0 = NdArray((3,4), 'f4')
    P1 = NdArray((8,9,10), 'f4')

    P = obj(P0, P1)
    r = gd(ε=0.001)(P)

    assert r.type[0] == obj(P0, P0) + obj(P1, P1)
    assert r.type[1] == obj(P0, P0) + obj(P1, P1)

def test_mse():
    P0 = NdArray((3,4), float)
    P1 = NdArray((8,9,10), float)
    P = obj(P0, P1)

    f = mse(P)

    assert f.type[0] == obj(P0, P0) + obj(P1, P1)
    assert f.type[1] == obj(P0, P0) + obj(P1, P1)
