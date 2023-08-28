from typing import Tuple, Any, Callable
from dataclasses import dataclass

# TODO: parametrise by array lib?
import numpy as np
import scipy.special as special

from yarrow import *
from catgrad.operation import Operation, LinearOperation

Shape = Tuple
Dtype = Any # TODO

# A value of type NdArray form the *objects* of our category -- the "types".
# An NdArrays is a pair of shape and dtype
@dataclass
class NdArray:
    shape: Shape
    dtype: Dtype

    @classmethod
    def from_ndarray(cls, x: np.ndarray):
        return NdArray(x.shape, x.dtype)

    # coproduct of shapes for convenience!
    def __add__(x, y):
        # allow e.g., NdArray((1,2,3), int) + 4 = NdArray((1,2,3,4), int)
        # and 0 + NdArray((1,2,3), int) = NdArray((0,1,2,3), int)
        if type(x) is NdArray and type(y) is int:
            return x + NdArray((y,), x.dtype)
        if type(y) is int and type(x) is NdArray:
            return x + NdArray((y,), x.dtype)

        if x.dtype != y.dtype:
            raise ValueError(f"tried to concatenate shapes {x} and {y} but dtypes differ")
        return NdArray(x.shape + y.shape, x.dtype)

wn0 = FiniteFunction.initial(None, dtype='object')
xn0 = FiniteFunction.initial(None, dtype='object')

def element(S: NdArray):
    """ Turn a shape into a type """
    return FiniteFunction(None, [S], dtype='object')

def obj(*args: List[NdArray]):
    """ Create an object of the category from a list of NdArray """
    return FiniteFunction(None, args, dtype='object')

# Parameters are like Constant but without values.
# They're used to make working with parametrised maps easier: if you try to
# reverse differentiate a circuit with Parameters in, you will get an error.
# Instead, you must first use `unparametrise`.
@dataclass
class Parameter(Operation):
    type: NdArray
    initializer: Callable[Any, np.ndarray] = None

    def __post_init__(self):
        pass

    @property
    def source(self):
        return wn0

    @property
    def target(self):
        return element(self.type)

    def fwd(self):
        return self.to_diagram()

    def residual(self):
        return wn0

    def rev(self):
        raise NotImplementedError("Don't try to reverse-differentiate Parameter operations!")

    def call(self, *args):
        raise NotImplementedError("Don't try to call Parameter operations!")

    def initialize(self):
        if self.initializer is not None:
            return self.initializer(self.type)
        return np.zeros(self.type.shape, self.type.dtype)

@dataclass
class Constant(LinearOperation):
    value: np.ndarray

    def __post_init__(self):
        if type(self.value) != np.ndarray:
            raise ValueError(f"Constant.value must be an ndarray, but was {type(self.value)}")

    @property
    def type(self):
        return NdArray(self.value.shape, self.value.dtype)

    @property
    def source(self):
        return wn0

    @property
    def target(self):
        return element(self.type)

    def dagger(self):
        return Discard(self.type).to_diagram()

    def call(self, *args):
        assert len(args) == 0
        return self.value

@dataclass
class Discard(LinearOperation):
    type: NdArray

    @property
    def source(self):
        return element(self.type)

    @property
    def target(self):
        return wn0

    def fwd(self):
        return self.to_diagram()

    def residual(self):
        return wn0

    # ! : A → I
    # R[!] : A×I → A
    def rev(self):
        v = np.zeros(self.type.shape, self.dtype)
        return Constant(v).to_diagram()

    def call(self, x):
        return None


@dataclass
class Add(LinearOperation):
    type: NdArray

    @property
    def source(self):
        return element(self.type) + element(self.type)

    @property
    def target(self):
        return element(self.type)

    def dagger(self):
        return Copy(self.type).to_diagram()

    def call(self, x, y):
        return x + y

@dataclass
class Negate(LinearOperation):
    type: NdArray

    @property
    def source(self):
        return element(self.type)

    @property
    def target(self):
        return element(self.type)

    def dagger(self):
        return self.to_diagram()

    def call(self, x):
        return -x


@dataclass
class Copy(LinearOperation):
    type: NdArray

    @property
    def source(self):
        return element(self.type)

    @property
    def target(self):
        return element(self.type) + element(self.type)

    def dagger(self):
        return Add(self.type).to_diagram()

    def call(self, x):
        return x, x

################################################################################
# Reshaping, Transposes, etc.

@dataclass
class Reshape(LinearOperation):
    X: Shape
    Y: Shape

    def __post_init__(self):
        # input and output must have same number of entries
        if np.prod(self.X.shape) != np.prod(self.Y.shape):
            raise ValueError("Must have np.prod(X) == np.prod(Y)")

    @property
    def source(self):
        return element(self.X)

    @property
    def target(self):
        return element(self.Y)

    def dagger(self) -> Diagram:
        return Reshape(self.Y, self.X).to_diagram()

    def call(self, x):
        assert x.shape == self.X.shape
        return x.reshape(self.Y.shape)

@dataclass
class Transpose2D(LinearOperation):
    X: NdArray
    Y: NdArray = None

    def __post_init__(self):
        if self.Y == None:
            self.Y = NdArray(tuple(reversed(self.X.shape)), self.X.dtype)

        if self.X.dtype != self.Y.dtype:
            raise ValueError("Transpose2D requires equal dtypes, but X.dtype = {self.X.dtype} and Y.dtype = {self.Y.dtype}")
        if len(self.X.shape) != 2 and len(self.Y.shape) != 2:
            raise ValueError(f"Transpose2D requires 2D shapes, but X.shape = {self.X.shape} and Y.shape = {self.Y.shape}")

    @property
    def source(self):
        return element(self.X)

    @property
    def target(self):
        return element(self.Y)

    def dagger(self) -> Diagram:
        return Transpose2D(Y, X).to_diagram()

    def call(self, x):
        assert x.shape == self.X.shape
        return x.T

################################################################################
# Nonlinear

@dataclass
class Multiply(Operation):
    # NOTE: Multiply is polymorphic; we allow broadcast multiplying scalars with
    # arrays to avoid having to create large ndarrays full of scalars.
    # TODO: Arguably this should be factored into a separate operation
    X: NdArray
    Y: NdArray

    def __post_init__(self):
        if self.X.dtype != self.Y.dtype:
            raise ValueError(f"Multiply operation requires matching dtypes, but {self.X.dtype} != {self.Y.dtype}")

        self.Z = self.X
        if self.X != self.Y:
            if sum(self.Y.shape) == 1:
                self.Z = self.X
            elif sum(self.X.shape) == 1:
                self.Z = self.Y
            else:
                raise ValueError("Multiply: if X.shape != Y.shape, then either X or Y must be a scalar")

    @property
    def source(self):
        return FiniteFunction(None, [self.X, self.Y], 'O')

    @property
    def target(self):
        return FiniteFunction(None, [self.Z], 'O')

    def fwd(self):
        return lens_fwd(self)

    def residual(self):
        return self.source

    def rev(self):
        X, Y, Z = element(self.X), element(self.Y), element(self.Z)

        mul0    = Multiply(self.Y, self.Z).to_diagram()
        mul1    = Multiply(self.X, self.Z).to_diagram()

        lhs = (twist(X, Y) @ copy(Z))
        mid = (identity(Y) @ twist(X, Z) @ identity(Z))
        rhs = mul0 @ mul1
        return lhs >> mid >> rhs

    def call(self, x, y):
        return x * y

@dataclass
class MatMul(Operation):
    X0: NdArray # input 1 (usually the parameter matrix)
    X1: NdArray # input 2 (usually a vector)

    def __post_init__(self):
        C, B0 = self.X0.shape
        B1, A = self.X1.shape

        # TODO: helpful error

        if B0 != B1:
            raise ValueError("MatMul requires compatible shapes, but got X0 = {self.X0.shape} and X1 = {self.X1.shape}")

        if self.X0.dtype != self.X1.dtype:
            raise ValueError("MatMul requires equal dtype, but got X0.dtype = {self.X0.type} and X1 = {self.X1.type}")

        self.Y = NdArray((C, A), dtype=self.X0.dtype)

        if self.X0.dtype != self.X1.dtype:
            raise ValueError(f"MatMul X0.dtype != X1.dtype with X0 = {self.X0}, X1 = {self.X1}")

    @property
    def source(self):
        return FiniteFunction(None, [self.X0, self.X1], 'O')

    @property
    def target(self):
        return FiniteFunction(None, [self.Y], 'O')

    def fwd(self):
        return lens_fwd(self.to_diagram())

    def residual(self):
        return self.source

    def rev(self):
        C, B = self.X0.shape
        _, A = self.X1.shape

        t1 = Transpose2D(self.X0).to_diagram()
        t2 = Transpose2D(self.X1).to_diagram()

        lhs = t1 @ t2 @ Copy(self.Y).to_diagram()

        permute = FiniteFunction(4, [2, 1, 0, 3])
        wn = t1.type[1] + t2.type[1] + self.target + self.target
        mid = Diagram.spider(FiniteFunction.identity(4), permute, wn, xn0)

        m1 = MatMul(self.Y, t2.type[1](0)).to_diagram()
        m2 = MatMul(t1.type[1](0), self.Y).to_diagram()

        return lhs >> mid >> (m1 @ m2)

    def call(self, M0, M1):
        assert M0.shape == self.X0.shape
        assert M1.shape == self.X1.shape
        return M0 @ M1

################################################################################
# Activation functions

@dataclass
class Sigmoid(Operation):
    type: NdArray

    @property
    def source(self):
        return obj(self.type)

    @property
    def target(self):
        return obj(self.type)

    def fwd(self):
        return lens_fwd(self.to_diagram())

    def rev(self):
        # Build the diagram for the expression
        #   σ(x) * (1 - σ(x)) * dy
        A = obj(self.type)

        mul = Multiply(self.type, self.type).to_diagram()
        id_A = identity(A)

        # 1 - σ(x)
        v = np.ones(self.type.shape, dtype=self.type.dtype)
        one_sub = (Constant(v).to_diagram() @ id_A) >> sub(A)

        # NOTE: only evaluates Sigmoid once; this is like a let binding:
        #   let x' = σ(x) in x' * (1 - x')
        top = self.to_diagram() >> copy(A) >> (one_sub @ id_A) >> mul

        # σ(x) * (1 - σ(x)) * dy
        return (top @ id_A) >> mul

    def residual(self):
        return obj(self.type)

    def call(self, x):
        return special.expit(x)

################################################################################
# Useful diagrams

def identity(w: FiniteFunction):
    return Diagram.identity(w, xn0)

def twist(x: FiniteFunction, y: FiniteFunction):
    return Diagram.twist(x, y, xn0)

def add(w: FiniteFunction):
    """ Create the canonical "add" morphism for any object w ∈ Σ₀* """
    if len(w) == 0:
        e = FiniteFunction.initial(None, 'O')
        return Diagram.empty(e, e)

    i = Diagram.half_spider(FiniteFunction.cointerleave(len(w)), w + w, xn0).dagger()
    f = Diagram.tensor_list([Add(t).to_diagram() for t in w.table])
    return i >> f

def negate(w: FiniteFunction):
    if len(w) == 0:
        e = FiniteFunction.initial(None, 'O')
        return Diagram.empty(e, e)

    return Diagram.tensor_list([Negate(t).to_diagram() for t in w.table])

def sub(w: FiniteFunction):
    return (identity(w) @ negate(w)) >> add(w)

def copy(w: FiniteFunction):
    """ Create the canonical "copy" morphism for any object w ∈ Σ₀* """
    if len(w) == 0:
        e = FiniteFunction.initial(None, 'O')
        return Diagram.empty(e, e)

    f = Diagram.tensor_list([Copy(t).to_diagram() for t in w.table])
    i = Diagram.half_spider(FiniteFunction.cointerleave(len(w)), w + w, xn0)
    return f >> i

def multiply(A: FiniteFunction, B: FiniteFunction):
    """ Create the canonical "multiplying" morphism for any object w ∈ Σ₀* """
    if len(A) != len(B):
        raise ValueError(f"Canonical multiply doesn't exist for len({A}) != len({B})")

    if len(A) == 0:
        e = FiniteFunction.initial(None, 'O')
        return Diagram.empty(e, e)

    i = Diagram.half_spider(FiniteFunction.cointerleave(len(A)), A + B, xn0).dagger()
    f = Diagram.tensor_list([Multiply(a, b).to_diagram() for a, b in zip(A.table, B.table)])
    return i >> f

def lens_fwd(c: Diagram):
    A = c.type[0]
    lhs = copy(A)
    rhs = c @ identity(A)
    return copy(A) >> (c @ identity(A))
