from typing import List
from dataclasses import dataclass
import ast

import numpy as np

import catgrad.signature as sig
from catgrad.compiler.decompose import SingletonOp

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
    return ast.Name(f"x{i}", ctx)

################################################################################

def discard(op: sig.Discard, x):
    # NOTE: this will kinda pollute the output a bit, but it's a nice "empty"
    # statement to use.
    return ast.Pass()

def _ndarray_to_list_ast(xs):
    """ Recursively convert a numpy ndarray to a python list """
    if xs.ndim == 1:
        # NOTE: we call tolist() here to conver to native python types.
        # If we don't do this, ast.Constant nodes will have e.g., np.float32
        # typed values, which will break later.
        values = xs.tolist()
        inner = [ ast.Constant(x) for x in values ]
    else:
        inner = [ ndarray_to_list_ast(x) for x in xs ]
    return ast.List(inner, ast.Load())

def constant(op: sig.Constant):
    # NOTE: the below is basically equivalent to this:
    # code = f"np.fromiter({str(op.value.tolist())}, dtype='{str(op.value.dtype)}')"
    # result = ast.parse(code, mode='eval').body
    fromiter = ast.Attribute(value=ast.Name(id="numpy", ctx=ast.Load()), attr='fromiter', ctx=ast.Load())
    list_literal = _ndarray_to_list_ast(op.value)
    dtype = ast.Constant(value=str(op.value.dtype))
    call = ast.Call(
        func=fromiter,
        args=[list_literal],
        keywords=[ast.keyword(arg='dtype', value=dtype)])
    return call

add = mk_binop(ast.Add())
negate = mk_unaryop(ast.USub())

def copy(op: sig.Copy, x):
    return ast.Tuple([x, x], ast.Load())

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
    # TODO: sigmoid implementation requires an import; what's a better way to
    # require the "special" module to be imported other than just hard-coding
    # the import?
    return ast.parse(f"scipy.special.expit({x.id})", mode="eval").body


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
    sig.Multiply: multiply,
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

def to_function(name: str, arg_ids: List[int], coarg_ids: List[int], op_statements: List[SingletonOp], decorator_list=[]):
    """ Create a Python AST node for a function definition from a list of operation assignments """
    args = ast.arguments(
            posonlyargs=[],
            args=[ ast.arg(mk_name(i, ast.Load()).id) for i in arg_ids ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[])

    coargs = [ mk_name(i, ast.Load()) for i in coarg_ids ]
    ret_val = ast.Tuple(coargs, ast.Load()) if len(coargs) > 1 else coargs[0]

    body = [ to_assignment(s.op, s.args, s.coargs) for s in op_statements ]
    # return ret_val if there's anything to return, otherwise Pass
    body += [ast.Return(ret_val) if len(coargs) > 0 else ast.Pass()]

    fn_def = ast.FunctionDef(name, args, body, decorator_list)
    return fn_def

def to_module(fn_def):
    imports = [
        # import scipy
        ast.Import(names=[ast.alias('scipy')]),
        ast.Import(names=[ast.alias('numpy')])
    ]
    body = imports
    body.append(fn_def)
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
