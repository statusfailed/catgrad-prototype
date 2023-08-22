from yarrow import *

from catgrad.signature import NdArray, element, xn0
import catgrad.signature as sig

from catgrad.initializer import normal

################################################################################
# Layers with parameters

def linear(source: NdArray, target: NdArray, initializer=normal()):
    """ A layer of type ``A → B``, which multiplies its A-dimensional vector
    input by a B×A matrix to obtain a B-dimensional output. """
    if source.dtype != target.dtype:
        raise ValueError("linear layer source and target must have the same dtype")

    # Matrix shape
    X0 = target + source
    X1 = source + 1
    matmul = sig.MatMul(X0, X1).to_diagram()
    param = sig.Parameter(target + source, initializer=initializer).to_diagram()
    reshape_A = sig.Reshape(source, source + 1).to_diagram()
    reshape_B = sig.Reshape(target + 1, target).to_diagram()
    return (param @ reshape_A) >> matmul >> reshape_B

def bias(T: NdArray):
    id_T  = Diagram.identity(element(T), xn0)
    param = sig.Parameter(T).to_diagram()
    add   = sig.Add(T).to_diagram()

    return (param @ id_T) >> add

def sigmoid(T: NdArray):
    return sig.Sigmoid(T).to_diagram()

def dense(source, target, activation=None, w_init=normal()):
    if activation is None:
        activation = lambda T: Diagram.identity(element(T), xn0)

    return linear(source, target, initializer=w_init) \
            >> bias(target) \
            >> activation(target)
