from hypothesis import given
from tests.strategies import shapes

from catgrad import *
from catgrad.signature import *
from catgrad.layer import *

@given(shapes())
def test_bias(S: Shape):
    T = (NdArray(S, dtype='int64'))
    c = bias(T)
    assert c.type[0] == FiniteFunction(None, [T], 'O')
    assert c.type[1] == FiniteFunction(None, [T], 'O')
