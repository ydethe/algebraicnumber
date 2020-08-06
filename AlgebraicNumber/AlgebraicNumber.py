import math

import numpy as np
from scipy.optimize import root
from numpy.polynomial.polynomial import Polynomial
from scipy import linalg as lin
from numpy.polynomial import polynomial as P

from AlgebraicNumber.utils import cpslq, simplify

# from AlgebraicNumber.inria_utils import inria_mul as an_mul
# from AlgebraicNumber.inria_utils import inria_add as an_add
from AlgebraicNumber.utils import kron_mul as an_mul
from AlgebraicNumber.utils import kron_add as an_add


class AlgebraicNumber(object):
    @classmethod
    def unity(cls):
        return AlgebraicNumber([-1, 1], 1.0)

    @classmethod
    def zero(cls):
        return AlgebraicNumber([0, 1], 0)

    @classmethod
    def constant(cls, a):
        b = -2 * np.real(a)
        c = np.abs(a) ** 2
        return AlgebraicNumber([c, b, 1], a)

    @classmethod
    def imaginary(cls):
        return AlgebraicNumber([1, 0, 1], 1j)

    def __init__(self, coeff, approx, _nosimp=False):
        self.coeff = coeff
        self.poly = Polynomial(coeff)
        self.approx = self.eval(approx)
        if not _nosimp:
            self._simplify()

    def _simplify(self):
        c = simplify(self.coeff, self.approx)

        self.coeff = c
        self.poly = Polynomial(c)

    def eval(self, approx=None):
        if approx is None:
            approx = self.approx

        def fun(X):
            x, y = X
            z = x + 1j * y
            P = self.poly(z)
            return [np.real(P), np.imag(P)]

        sol = root(fun, x0=[np.real(approx), np.imag(approx)])
        if sol.success:
            x, y = sol.x
            z = x + 1j * y
            return z
        else:
            print(sol)
            raise ValueError

    def inverse(self):
        coeff = self.coeff

        res = AlgebraicNumber(coeff[-1::-1], 1 / self.approx)

        return res

    def __mul__(self, b):
        """
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2*sqrt_3
        >>> p.coeff
        array([-6,  0,  1]...
        
        """
        Q = an_mul(self.coeff, b.coeff)

        res = AlgebraicNumber(Q, self.approx * b.approx)

        return res

    def __truediv__(self, b):
        """
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2/sqrt_3
        >>> p.coeff
        array([-2,  0,  3]...
        
        """
        ib = b.inverse()
        return self * ib

    def __neg__(self):
        """
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> p = -sqrt_2
        >>> p.coeff
        array([-2,  0,  1]...
        
        """
        n = len(self.coeff)
        coeff = np.array(self.coeff)

        R = coeff * (-1) ** np.arange(n)

        res = AlgebraicNumber(list(R), -self.approx)

        return res

    def __sub__(self, b):
        nb = -b
        return self + nb

    def __add__(self, b):
        """
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2+sqrt_3
        >>> p.coeff
        array([  1,   0, -10,   0,   1]...
        >>> ref = np.sqrt(2) + np.sqrt(3)
        >>> np.abs(p.approx - ref) < 1e-10
        True
        
        """
        Q = an_add(self.coeff, b.coeff)

        res = AlgebraicNumber(Q, self.approx + b.approx)

        return res

    def conj(self):
        """
        
        >>> z = AlgebraicNumber.unity() + AlgebraicNumber.imaginary()
        >>> z.coeff
        array([ 2, -2,  1]...
        >>> p = z*z.conj()
        >>> p.coeff
        array([-2,  1]...
        
        """
        coeff = self.coeff

        res = AlgebraicNumber(coeff, np.conj(self.approx))

        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
