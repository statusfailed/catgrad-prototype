import ast
import numpy as np

from catgrad.compiler.python import *
from catgrad.compiler.decompose import SingletonOp

# pretty-print an AST
# NOTE: fix_missing_locations is 
def show(expr):
    print(ast.unparse(ast.fix_missing_locations(expr)))

def test_to_assignment():
    T = sig.NdArray((3,4), dtype='u1')
    U = sig.NdArray((4,3), dtype='u1')
    V = sig.NdArray((3*4,), dtype='u1')
    W = sig.NdArray((4,1), dtype='u1')

    c = np.array([1, 2, 3], dtype='u1')

    # check that all operations can be converted to an assignment statement
    show(to_assignment(sig.Constant(c), [], [0]))
    show(to_assignment(sig.Discard(T), [1], []))
    show(to_assignment(sig.Add(T), [1,2], [0]))
    show(to_assignment(sig.Negate(U), [0], [0]))
    show(to_assignment(sig.Copy(U), [0], [1, 2]))
    show(to_assignment(sig.Reshape(T, U), [0], [1]))
    show(to_assignment(sig.Transpose2D(T, V), [0], [0]))
    show(to_assignment(sig.MatMul(T, W), [1,2], [0]))
    show(to_assignment(sig.Sigmoid(T), [0], [0]))

def test_to_function():
    # Test a simple example program, shown as a string diagram below:
    #
    #       x0 --|---|
    #            | + |--x2--[ negate ]--- x3
    #       x1 --|---|
    #
    T = sig.NdArray((3,4), dtype='u1')
    statements = [
        SingletonOp(sig.Add(T), [0, 1], [2]),
        SingletonOp(sig.Negate(T), [2], [3]),
    ]
    fn_def = to_function('f', [0, 1], [2], statements)

    show(fn_def)


def _compile_ast(tree, module_name='test_module'):
    code = compile(tree, module_name, mode='exec')
    scope = {}
    exec(code, scope)
    return scope['foo']

def test_to_module():
    SOURCE_CODE = """
import scipy
from numpy import fromiter as array

def foo(x0, x1):
    x2 = x0 + x1
    x3 = scipy.special.expit(x2)
    x4, x5 = x3, x3
    return x4, x5
    """

    # NOTE: type information is actually erased in compilation, but we could put
    # in assertions to do runtime checking
    T = sig.NdArray((3,4), dtype='u1')
    statements = [
        SingletonOp(sig.Add(T), [0, 1], [2]),
        SingletonOp(sig.Sigmoid(T), [2], [3]),
        SingletonOp(sig.Copy(T), [3], [4,5]),
    ]
    fn_def = to_function('foo', [0, 1], [4,5], statements)

    parsed_tree = ast.parse(SOURCE_CODE)
    parsed_function = _compile_ast(parsed_tree)

    tree = ast.fix_missing_locations(to_module(fn_def))
    function = _compile_ast(tree)

    # test some made up values
    values = [(1.0, 0.0), (1.0, 2.0), (10.0, 4.0)]
    for x0, x1 in values:
        assert function(x0, x1) == parsed_function(x0, x1)
