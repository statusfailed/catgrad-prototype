from hypothesis import given

from tests.strategies import span

import numpy as np
from yarrow import *

import catgrad.signature as sig
from catgrad.signature import NdArray, element
from catgrad import signature

import catgrad.parameters as parameters
from catgrad.parameters import factor_parameters
from catgrad.layer import linear, bias, dense

# given a finite function f and predicate p, check that filter(f, p) filters f by p.
@given(s=span(left=2))
def test_filter_by_length(s):
    p, f = s
    r = parameters.filter_by(f, p)
    assert len(r.table) == p.table.sum()

# TODO: make this a hypothesis test!
def test_singleton_filter_operations():
    xn = FiniteFunction(None, 'x', 'O')
    a = FiniteFunction(None, ['A', 'B'], 'O')
    b = FiniteFunction(None, ['C'], 'O')
    c = Diagram.singleton(a, b, xn)
    assert c.type == (a, b)

    p = FiniteFunction.terminal(c.G.X).inject0(1)
    d = parameters.filter_operations(c, p)
    assert d.G.W == c.G.W
    assert d.G.Ei == 0
    assert d.G.Eo == 0
    assert d.G.X == 0

# TODO: make this a hypothesis test!
def test_singleton_tensor_filter_operations():
    xn1 = FiniteFunction(None, 'x', 'O')
    a = FiniteFunction(None, ['A', 'B'], 'O')
    b = FiniteFunction(None, ['C'], 'O')
    c1 = Diagram.singleton(a, b, xn1)
    assert c1.type == (a, b)

    xn2 = FiniteFunction(None, 'y', 'O')
    c2 = Diagram.singleton(b, a, xn2)
    assert c2.type == (b, a)

    c = c1 @ c2

    # [ 0 1 ] => delete the 0th operation, which is 'x'
    p = FiniteFunction.terminal(1) @ FiniteFunction.terminal(1)
    d = parameters.filter_operations(c, p)

    assert d.G.W == c.G.W
    assert d.G.Ei == 1
    assert d.G.Eo == 2
    assert d.G.X == 1

# TODO: make this a hypothesis test!
def test_singleton_compose_filter_operations():
    xn1 = FiniteFunction(None, 'x', 'O')
    a = FiniteFunction(None, ['A', 'B'], 'O')
    b = FiniteFunction(None, ['C'], 'O')
    c1 = Diagram.singleton(a, b, xn1)
    assert c1.type == (a, b)

    xn2 = FiniteFunction(None, 'y', 'O')
    c2 = Diagram.singleton(b, a, xn2)
    assert c2.type == (b, a)

    c = c1 >> c2

    # [ 0 1 ] => delete the 0th operation, which is 'x'
    p = FiniteFunction.terminal(1) @ FiniteFunction.terminal(1)
    d = parameters.filter_operations(c, p)

    assert d.G.W == c.G.W
    assert d.G.Ei == 1
    assert d.G.Eo == 2
    assert d.G.X == 1

# When we unparametrise something with only Parameter nodes, we should get a map
# with no operations whose input is the target of all those parameter nodes.
def test_factor_parameters_single_param():
    T = NdArray((3,2), int)
    c = signature.Parameter(T).to_diagram()

    A = FiniteFunction(None, [], 'object')
    B = FiniteFunction(None, [T], 'object')
    assert c.type == (A, B)

    p, f = factor_parameters(c)
    assert f.type == (B, B)
    assert len(p.type[0]) == 0
    assert p.type[1] == element(T)

    assert ((p @ sig.identity(A)) >> f) == c

def test_factor_parameters_linear():
    X = NdArray((2,), dtype=int)
    Y = NdArray((3,), dtype=int)
    c = linear(X, Y)

    P = FiniteFunction(None, [Y+X], 'object')
    A = FiniteFunction(None, [X], 'object')
    B = FiniteFunction(None, [Y], 'object')

    assert c.type == (A, B)

    p, f = factor_parameters(c)
    assert f.type == (P + A, B)
    assert len(p.type[0]) == 0
    assert p.type[1] == P

    assert ((p @ sig.identity(A)) >> f) == c

def test_factor_parameters_bias():
    T = NdArray((3, 2, 1), 'int64')
    c = bias(T)

    A = FiniteFunction(None, [T], 'object')
    P = A

    assert c.type == (A, A)

    p, f = factor_parameters(c)
    assert f.type == (A + A, A)
    assert len(p.type[0]) == 0
    assert p.type[1] == P

    assert ((p @ sig.identity(A)) >> f) == c

def test_factor_parameters_dense():
    X = NdArray((4,), int)
    Y = NdArray((3,), int)
    c = dense(X, Y, activation=None)

    P = FiniteFunction(None, [Y+X, Y], 'O')
    A = FiniteFunction(None, [X], 'O')
    B = FiniteFunction(None, [Y], 'O')

    assert c.type == (A, B)

    p, f = factor_parameters(c)
    assert f.type == (P + A, B)
    assert len(p.type[0]) == 0
    assert p.type[1] == P
