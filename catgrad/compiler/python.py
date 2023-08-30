from typing import List
from dataclasses import dataclass
import ast
import catgrad.signature as sig

def mk_binop(ast_op):
    def binop(op: sig.Operation, x1, x2):
        return ast.BinOp(x1, ast_op, x2)
    return binop

def mk_unaryop(ast_op):
    def unaryop(op: sig.Operation, x):
        return ast.UnaryOp(ast_op, x)
    return unaryop

def mk_name(i: int, ctx):
    """ Create a name (e.g. ``x0``) from an integer (``0``) and a context
    (``ast.Load()`` or ``ast.Store()``) """
    return ast.Name(f"x{i}", ast.Load())

################################################################################

def discard(op: sig.Discard, x):
    # NOTE: this will kinda pollute the output a bit, but it's a nice "empty"
    # statement to use.
    return ast.Pass()

def constant(op: sig.Constant):
    return ast.Constant(op.value)

add = mk_binop(ast.Add())
negate = mk_unaryop(ast.USub())

def copy(op: sig.Copy, x):
    return ast.Tuple([x, x], ast.Load()) # TODO?

def reshape(op: sig.Reshape, x):
    return ast.parse(f"{x.id}.reshape({op.Y.shape})", mode="eval").body

def transpose2d(op: sig.Transpose2D, x):
    return ast.Attribute(value=x, attr='T', ctx=ast.Load())

multiply = mk_binop(ast.Mult())
matmul = mk_binop(ast.MatMult())

def sigmoid(op: sig.Sigmoid, x):
    # NOTE: ast.parse with mode='eval' returns an Expression, but we only want the "body" of the expression.
    # If we try to assign an expression directly, it eliminates the LHS vars for some reason.
    # TODO: why?
    return ast.parse(f"special.expit({x.id})", mode="eval").body


# TODO: this is kinda dumb because we could have just made an "ast" method on
# each op which returns a python AST.  However, if we want to compile to other
# backends, we're going to need different implementations, and we (probably?)
# don't want to put them all in a big god-object...?
OP_TO_AST = {
    sig.Constant: constant,
    sig.Discard: discard,
    sig.Add: add,
    sig.Negate: negate,
    sig.Copy: copy,
    sig.Reshape: reshape,
    sig.Transpose2D: transpose2d,
    sig.MatMul: matmul,
    sig.Sigmoid: sigmoid,
}

def to_assignment(op: sig.Operation, arg_ids: List[int], coarg_ids: List[int], op_to_ast=OP_TO_AST):
    fn = op_to_ast[type(op)]
    args = [ mk_name(i, ast.Load()) for i in arg_ids ]
    coargs = [ mk_name(i, ast.Store()) for i in coarg_ids ]

    # assign multiple vars if coarity > 1
    if len(coargs) > 1:
        coargs = [ast.Tuple(coargs, ast.Store())]

    expr = fn(op, *args)
    assign = ast.Assign(targets=coargs, value=expr)
    return assign

# TODO: produce a list of OpStatement from a Diagram.
@dataclass
class OpStatement:
    op: sig.Operation
    args: List[int]
    coargs: List[int]

def to_function(name, arg_ids, coarg_ids, op_statements: List[OpStatement], decorator_list=[]):
    args = ast.arguments(
            posonlyargs=[],
            args=[ ast.arg(mk_name(i, ast.Load()).id) for i in arg_ids ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[])

    coargs = [ mk_name(i, ast.Store()) for i in coarg_ids ]
    ret_val = ast.Tuple(coargs, ast.Load()) if len(coargs) > 1 else coargs

    body = [ to_assignment(s.op, s.args, s.coargs) for s in op_statements ]
    # return ret_val if there's anything to return, otherwise Pass
    body += [ast.Return(ret_val) if len(coargs) > 0 else ast.Pass()]

    fn_def = ast.FunctionDef(name, args, body, decorator_list)
    return fn_def

################################################################################

def show(expr):
    print(ast.unparse(ast.fix_missing_locations(expr)))

def test_to_function():
    # example program:
    #   x0 --|---|
    #        | + |--x2--[ negate ]--- x3
    #   x1 --|---|

    T = sig.NdArray((3,4), dtype='u1')
    statements = [
        OpStatement(sig.Add(T), [0, 1], [2]),
        OpStatement(sig.Negate(T), [2], [3]),
    ]
    fn_def = to_function('f', [0, 1], [2], statements)

    show(fn_def)

def test_to_assignment():
    T = sig.NdArray((3,4), dtype='u1')
    U = sig.NdArray((4,3), dtype='u1')
    V = sig.NdArray((3*4,), dtype='u1')
    W = sig.NdArray((4,1), dtype='u1')

    c = np.array([1, 2, 3], dtype='u1')

    show(to_assignment(sig.Constant(c), [], [0]))
    show(to_assignment(sig.Discard(T), [1], []))
    show(to_assignment(sig.Add(T), [1,2], [0]))
    show(to_assignment(sig.Negate(U), [0], [0]))
    show(to_assignment(sig.Copy(U), [0], [1, 2]))
    show(to_assignment(sig.Reshape(T, U), [0], [1]))
    show(to_assignment(sig.Transpose2D(T, V), [0], [0]))
    show(to_assignment(sig.MatMul(T, W), [1,2], [0]))
    show(to_assignment(sig.Sigmoid(T), [0], [0]))

if __name__ == "__main__":
    import numpy as np
    # test_to_assignment()
    test_to_function()
