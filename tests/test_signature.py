from hypothesis import given
from tests.strategies import shapes

from catgrad import *
from catgrad.signature import *

def test_add_empty():
    c = add(FiniteFunction.initial(0))
    assert c.type[0] == FiniteFunction.initial(None, 'O')
    assert c.type[1] == FiniteFunction.initial(None, 'O')

def test_add_triple():
    T, U, V = NdArray((3,), 'u8'), NdArray((2,), 'u4'), NdArray((1,), 'u2')
    A = FiniteFunction(None, [T, U, V], 'O')
    c = add(A)

    assert c.type[0] == A + A
    assert c.type[1] == A

def test_multiply_triple():
    T, U, V = NdArray((3,), 'u8'), NdArray((2,), 'u4'), NdArray((1,), 'u2')
    A = FiniteFunction(None, [T, U, V], 'O')
    c = multiply(A, A)

    assert c.type[0] == A + A
    assert c.type[1] == A

def test_multiply_constant():
    T, U = NdArray((3,), float), NdArray((2,), float)
    C    = NdArray((1,), float)

    A = FiniteFunction(None, [T, U], 'O')
    B = FiniteFunction(None, [C, C], 'O')

    c = multiply(A, B)
    assert c.type[0] == A + B
    assert c.type[1] == A

    c = multiply(B, A)
    assert c.type[0] == B + A
    assert c.type[1] == A

def test_copy_triple():
    T, U, V = NdArray((3,), 'u8'), NdArray((2,), 'u4'), NdArray((1,), 'u2')

    A = FiniteFunction(None, [T, U, V], 'O')
    c = copy(A)

    assert c.type[0] == A
    assert c.type[1] == A + A

def test_multiply_rev():
    A = NdArray((4,3,2), 'u8')
    T = element(A)

    r = Multiply(A, A).rev()

    assert r.type[0] == (T + T) + T
    assert r.type[1] == (T + T)

def test_transpose2d():
    T = NdArray((3,2), float)

    c = Transpose2D(T)
    assert c.target == element(NdArray((2,3), float))

def test_matmul():
    X0 = NdArray((3,4), float)
    X1 = NdArray((4,1), float)
    Y  = NdArray((3,1), float)

    m = MatMul(X0, X1).to_diagram()

    assert m.type[0] == obj(X0, X1)
    assert m.type[1] == obj(Y)

def test_matmul_rev():
    X0 = NdArray((3,4), float)
    X1 = NdArray((4,1), float)
    Y  = NdArray((3,1), float)

    r = MatMul(X0, X1).rev()

    assert r.type[0] == obj(X0, X1) + obj(Y)
    assert r.type[1] == obj(X0, X1)
