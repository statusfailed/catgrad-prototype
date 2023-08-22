""" Initializers are functions which take an NdArray (a shape and dtype), and
return an initial parameter value (usually random) """
import numpy as np

import catgrad.signature as sig

def normal(mean=0, stddev=0.01):
    def normal_init(type: sig.NdArray):
        return np.random.normal(mean, stddev, type.shape).astype(type.dtype)
    return normal_init
