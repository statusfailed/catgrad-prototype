from catgrad.signature import NdArray
from catgrad.layer import linear, dense, sigmoid
from catgrad.compile import optic_to_python
from catgrad.learner import gd, mse
from catgrad.parameters import factor_parameters
import catgrad.optic as optic

import argparse
import numpy as np
import pandas as pd
import scipy.special as special

################################################################################
# Compiling the model

def make_learner(model, update, displacement):
    p, f = factor_parameters(model)

    A, B = model.type
    P = p.type[1]

    id_A = optic.identity(A)
    u = update(P)
    d = displacement(B)
    of = optic.Optic().map_arrow(f)

    step = (u @ id_A) >> of >> d
    return P, A, B, p, step

def get_step(model):
    P, A, B, p, f = make_learner(model, lambda P: gd(P, ε=0.01), mse)
    adapted = optic.adapt_optic(f)
    code = f"""
from numpy import float32, dtype
from catgrad.signature import *
{optic_to_python(adapted, model_name="step")}"""

    print(code)
    scope = {}
    exec(code, scope)
    step = scope['step']

    def wrapped_step(*args):
        result = step(*args)
        y = result[:len(B)]
        p = result[len(B):len(B) + len(P)]
        x = result[:-len(A)]
        return y, p, x

    # TODO: this is a hack: we *assume* the p morphism has its operations in the
    # same order as they appear in the boundary, but this will break easily if
    # factor_parameters changes!
    theta = [ x.initialize() for x in p.G.xn.table ]
    return theta, wrapped_step

INPUT_TYPE = NdArray((4,), 'f4')
OUTPUT_TYPE = NdArray((3,), 'f4')
HIDDEN_TYPE = NdArray((20,), 'f4')

################################################################################
# Iris data loading etc.

def accuracy(y_pred, y_true):
    num = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
    den = len(y_true)
    return np.sum(num) / den

def load_iris(path):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    return train_input, train_labels

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iris-data', default='data/iris.csv')
    parser.add_argument('model', default='linear')
    args = parser.parse_args()

    if args.model == 'linear':
        model = linear(INPUT_TYPE, OUTPUT_TYPE)
    elif args.model == 'simple':
        model = linear(INPUT_TYPE, OUTPUT_TYPE) >> sigmoid(OUTPUT_TYPE)
    elif args.model == 'dense':
        model = dense(INPUT_TYPE, OUTPUT_TYPE, activation=sigmoid)
    elif args.model == 'hidden':
        model = dense(INPUT_TYPE, HIDDEN_TYPE, activation=sigmoid) \
                >> dense(HIDDEN_TYPE, OUTPUT_TYPE, activation=sigmoid)

    # compile the inner step of the training loop
    # NOTE: parameters are auto-initialized to zeros (see get_step)
    p, step  = get_step(model)

    # Load data from CSV
    train_input, train_labels = load_iris(args.iris_data)
    N = len(train_input)

    x = train_input
    y = train_labels

    # NOTE: Adding +75 gets a big accuracy boost - this basically performs the
    # same function as randomising the data.
    NUM_ITER = 150 * 400

    q = None
    for i in range(0, NUM_ITER):
        # reorder every epoch
        if i % len(x) == 0:
            q = np.random.permutation(len(x))
        xi = x[q[i % len(x)]]
        yi = y[q[i % len(y)]]

        yhat, p, _ = step(*p, xi, yi)

    y_hats = np.zeros_like(y)
    for i in range(0, len(x)):
        # NOTE: using "step" to predict like this works, but it's doing a lot of
        # extra redundant computation too.
        [y_hat], _, _ = step(*p, x[i], y[i])
        y_hats[i] = y_hat

    print('p: ', p)
    print('ŷ₀ = ', y_hats[0].T)
    print('y₀ = ', y[0].T)

    print(f'accuracy: {100*accuracy(y_hats, y)}%')

if __name__ == "__main__":
    main()
