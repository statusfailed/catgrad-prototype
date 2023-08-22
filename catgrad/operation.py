from abc import ABC, abstractmethod
from yarrow import *

class Operation(ABC):
    """ A base class for operations in the signature """
    @property
    @abstractmethod
    def source(self):
        ...

    @property
    @abstractmethod
    def target(self):
        ...

    @abstractmethod
    def fwd(self):
        ...

    @abstractmethod
    def rev(self):
        ...

    @abstractmethod
    def residual(self):
        ...

    # NOTE: this method will be wrapped by __call__.
    @abstractmethod
    def call(self, *args):
        ...

    def __call__(self, *args):
        nargs = len(args)
        arity = len(self.source)
        coarity = len(self.target)

        if nargs != arity:
            raise ValueError(f"Operation {self} has arity {arity} but it was called with {nargs} arguments!")

        result = self.call(*args)
        if coarity == 0:
            assert result == None
        elif coarity > 1:
            assert type(result) is tuple
            # NOTE: in contrast to arity errors, this is the fault of the
            # implementor of the Operation!
            if len(result) != coarity:
                raise ValueError(f"Operation {self} has coarity {coarity} but it returned {nargs} arguments!")

        return result

    def to_diagram(self):
        xn = FiniteFunction(None, [self], dtype='object')
        return Diagram.singleton(self.source, self.target, xn)

class LinearOperation(Operation):
    """ A LinearOperation is an Operation whose forward map is op.to_diagram(), and whose reverse map is op.dagger() """

    def fwd(self):
        return self.to_diagram()

    def rev(self):
        return self.dagger()

    def residual(self):
        # TODO: this assumes the signature is made of objects
        return FiniteFunction.initial(None, dtype='object')

    @abstractmethod
    def dagger(self) -> Diagram:
        ...

    @property
    @abstractmethod
    def source(self):
        ...

    @property
    @abstractmethod
    def target(self):
        ...

    @abstractmethod
    def call(self, *args):
        ...
