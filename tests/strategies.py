import numpy as np
import hypothesis.strategies as st

from tests.finite_function_strategies import *

# we don't really want to be generating huge multidimensional ndarrays during
# tests, so this is a small number!
MAX_DIMENSIONS = 3
dimensions = st.integers(min_value=0, max_value=MAX_DIMENSIONS)

# max number of entries per dimension
MAX_SIZE = 10

@st.composite
def shapes(draw):
    d = draw(dimensions)
    return tuple(np.random.randint(0, MAX_SIZE, d))

# Draw
#   f : A → B
#   p : A → 2
@st.composite
def span(draw, left=None, apex=None, right=None):
    (A, L) = draw(arrow_type(apex, left))
    (A, R) = draw(arrow_type(apex, right))

    l = draw(finite_functions(source=A, target=L))
    r = draw(finite_functions(source=A, target=R))

    return l, r
