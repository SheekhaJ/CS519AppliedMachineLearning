'''sparse vector based on defaultdict'''

__author__ = "lhuang"

from collections import defaultdict

class svector(defaultdict):

    def __init__(self, old=None):
        if old is not None:
            defaultdict.__init__(self, float, old)
        else:
            defaultdict.__init__(self, float)

    def __iadd__(self, other): # a += b
        for k, v in other.items():
            self[k] += v
        return self

    def __add__(self, other): # a + b
        new = svector()
        for k, v in self.items():
            new[k] = v
        for k, v in other.items():
            new[k] += v
        return new

    def __sub__(self, other): # a - b
        return self + (-1) * other

    def __isub__(self, other): # a -= b
        self += (-1) * other

    def __mul__(self, c): # v * c where c is scalar
        new = svector()
        for k, v in self.items():
            new[k] = v * c
        return new

    __rmul__ = __mul__ # c * v where c is scalar

    def dot(self, other): # dot product
        a, b = (self, other) if len(self) < len(other) else (other, self) # fast
        return sum(v * b[k] for k, v in a.items())

    def __neg__(self): # -a
        new = svector()
        for k, v in self.items():
            new[k] = -v
        return new

    def copy(self):
        return svector(self)
        
