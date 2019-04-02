from functools import reduce
from operator import mul
import sys


class ConjunctionFunction:
    def __init__(self, f):
        if not callable(f):
            raise ValueError(type(f), " is not a function")
        self.f = f


def validate_conjunction(conj):
    """
    Check if conj is a ConjunctionFunction, if it's not then try to load from the CONJ_DICT.
    :param conj: conjunction function
    """
    # Check whether the user provided a properly formatted conjunction function
    if isinstance(conj, ConjunctionFunction):
        return conj
    else:
        # Otherwise check whether the specified conjunction function exists
        try:
            return CONJ_DICT[conj]
        except KeyError:
            sys.exit("Conjunction function undefined")


CONJ_DICT = {"sum": ConjunctionFunction(lambda l: sum(l)),
             "prod": ConjunctionFunction(lambda l: reduce(mul, l)),
             "max": ConjunctionFunction(lambda l: max(l)),
             "min": ConjunctionFunction(lambda l: min(l)),
             "avg": ConjunctionFunction(lambda l: sum(l)/len(l)),
             "or": ConjunctionFunction(lambda l: l[0] + l[1] - (l[0] * l[1]) if len(l) == 2 else -1)}
