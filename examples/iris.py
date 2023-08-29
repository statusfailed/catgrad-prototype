# from catgrad import NdArray
# from catgrad.layer import linear, dense, sigmoid
# from catgrad.learner import get_step, gd, mse
from catgrad import NdArray, layer, learner, compile, compile_model

import argparse
import numpy as np
import pandas as pd
import scipy.special as special

INPUT_TYPE = NdArray((4,), 'f4')
OUTPUT_TYPE = NdArray((3,), 'f4')
HIDDEN_TYPE = NdArray((20,), 'f4')

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
        model = layer.linear(INPUT_TYPE, OUTPUT_TYPE)
    elif args.model == 'simple':
        model = layer.linear(INPUT_TYPE, OUTPUT_TYPE) >> layer.sigmoid(OUTPUT_TYPE)
    elif args.model == 'dense':
        model = layer.dense(INPUT_TYPE, OUTPUT_TYPE, activation=layer.sigmoid)
    elif args.model == 'hidden':
        model = layer.dense(INPUT_TYPE, HIDDEN_TYPE, activation=layer.sigmoid) \
                >> layer.dense(HIDDEN_TYPE, OUTPUT_TYPE, activation=layer.sigmoid)

    # compile the inner step of the training loop
    # NOTE: parameters are auto-initialized to zeros (see get_step)
    p, step, predict, code = compile_model(model, learner.gd(ε=0.01), learner.mse)
    # print(code)

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

    # Predict, using the (faster) model circuit
    print("compiling predict...")
    predict, _ = compile(predict, function_name='predict')
    print("predicting...")
    y_hats = np.zeros_like(y)
    for i in range(0, len(x)):
        # NOTE: using "step" to predict like this works, but it's doing a lot of
        # extra redundant computation too.
        y_hats[i] = predict(*p, x[i])
        # you can also compute predictions using "step", but it's slower.
        # [y_hats[i]], _, _ = step(*p, x[i], y[i])

    print(f'accuracy: {100*accuracy(y_hats, y)}%')

if __name__ == "__main__":
    main()
