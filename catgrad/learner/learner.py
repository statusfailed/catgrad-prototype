from catgrad.parameters import factor_parameters
import catgrad.optic as optic

def make_learner(model, update, displacement):
    """ Assemble a learner from model, update, and displacement maps.

    * model is a map ``f : A → B`` containing Parameter operations of type ``P``
    * update is a function producing an update map ``update(P)`` from a parameter type ``P``
    * displacement is a function producing an displacement map ``displacement(B)`` from a type ``B``
    """
    p, f = factor_parameters(model)

    A, B = model.type
    P = p.type[1]

    id_A = optic.identity(A)
    u = update(P)
    d = displacement(B)
    of = optic.Optic().map_arrow(f)

    step = (u @ id_A) >> of >> d
    return p, step, (P, A, B)
