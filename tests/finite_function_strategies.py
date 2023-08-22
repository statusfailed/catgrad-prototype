# NOTE: this code is duplicated from yarrow! Export it!
import numpy as np
import hypothesis.strategies as st

from yarrow import *

_MAX_OBJECTS = 32

# a generator for objects of FinFun
objects = st.integers(min_value=0, max_value=_MAX_OBJECTS)
nonzero_objects = st.integers(min_value=1, max_value=32)

def _is_valid_arrow_type(s, t):
    if t == 0:
        return s == 0
    return True

@st.composite
def arrow_type(draw, source=None, target=None):
    """ Generate a random type of finite function.
    For example, a type of n → 0 is forbidden.
    """
    # User specified both target and source
    if target is not None and source is not None:
        if target == 0 and source != 0:
            raise ValueError("No arrows exist of type n → 0 for n != 0.")
        return source, target

    elif source is None:
        # any target
        target = draw(objects) if target is None else target

        if target == 0:
            source = 0
        else:
            source = draw(objects)

        return source, target

    # target is None, but source is not
    target = draw(nonzero_objects) if source > 0 else draw(objects)
    return source, target

# generate a random FiniteFunction
@st.composite
def finite_functions(draw, source=None, target=None):
    source, target = draw(arrow_type(source=source, target=target))
    assert _is_valid_arrow_type(source, target)

    # generate a random array of elements in {0, ..., target - 1}
    if target == 0:
        # FIXME: remove np hardcoding for other backends.
        table = np.zeros(0, dtype=int)
    else:
        table = np.random.randint(0, high=target, size=source)

    return FiniteFunction(target, table)
