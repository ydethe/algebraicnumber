import math

import numpy as np
from scipy.optimize import root
from scipy import linalg as lin

from AlgebraicNumber.QPolynomial import QPolynomial
from AlgebraicNumber.utils import *
from AlgebraicNumber.inria_utils import inria_mul as an_mul
from AlgebraicNumber.inria_utils import inria_add as an_add


class AlgebraicNumber(object):
    """

    Examples:
      >>> a = AlgebraicNumber.imaginary()

    """

    @classmethod
    def unity(cls):
        return AlgebraicNumber([-1, 1], 1.0)

    @classmethod
    def zero(cls):
        return AlgebraicNumber([0, 1], 0)

    @classmethod
    def integer(cls, a):
        return AlgebraicNumber([-a, 1], a)

    @classmethod
    def imaginary(cls):
        return AlgebraicNumber([1, 0, 1], 1j)

    def __init__(self, coeff, approx, _nosimp=False):
        if isinstance(coeff, QPolynomial):
            self.poly = coeff
        else:
            self.poly = QPolynomial(coeff=coeff)

        self.approx = self.eval(approx)
        if not _nosimp:
            self._simplify()

    def _simplify(self):
        self.poly = simplify(self.poly, self.approx)

    @property
    def coeff(self):
        p = self.poly.getCoefficients()
        return p

    def eval(self, approx=None):
        if approx is None:
            approx = self.approx

        r = self.poly.roots()
        k = np.argmin(np.abs(r - approx))

        return r[k]

    def plotRoots(self, axe=None, **kwargs):
        """Plots the roots of the minimal polynomial of the number

        Args:
          axe
            If given, a matplotlib axe to draw on. By default, plotRoots creates it
          kwargs
            List of arguments to format the plot. Must not specify linestyle.

        """
        if axe is None:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.grid(True)
            show = True
        else:
            show = False

        if not "marker" in kwargs.keys():
            kwargs["marker"] = "o"

        rc = self.poly.roots()
        axe.plot(np.real(rc), np.imag(rc), linestyle="", **kwargs)
        axe.plot(
            [np.real(self.approx)], [np.imag(self.approx)], linestyle="", marker="*"
        )

        if show:
            plt.show()

    def pow(self, p: int, q: int = 1) -> "AlgebraicNumber":
        r"""If :math:`\alpha` is an alebraic number, computes

        .. math::
            \alpha^{p/q}

        Examples:
          >>> a = AlgebraicNumber([-2, 0, 1], 1.414)
          >>> a.pow(2).poly.printCoeff()
          '[-2,1]'
          >>> a.pow(1,2).poly.printCoeff()
          '[-2,0,0,0,1]'

        """
        if q == 0:
            raise ZeroDivisionError

        res = AlgebraicNumber.unity()
        for i in range(p):
            res = res * self

        h = res.poly
        Xq = QPolynomial(coeff=[0] * q + [1])

        res = h.compose(Xq)

        res = AlgebraicNumber(res, self.approx ** (p / q))

        return res

    def inverse(self) -> "AlgebraicNumber":
        ZERO = AlgebraicNumber.zero()
        if self == ZERO:
            raise ZeroDivisionError

        p = self.poly.getCoefficients()

        res = AlgebraicNumber(p[-1::-1], 1 / self.approx)

        return res

    def __eq__(self, b):
        sep_a = self.poly.mahler_separation_bound()
        sep_b = b.poly.mahler_separation_bound()
        eps = min(sep_a, sep_b) / 2
        eq = np.abs(self.approx - b.approx) < eps

        return eq

    def __neq__(self, b):
        return not self.__eq__(b)

    def __repr__(self):
        s = str(self.poly)
        elem = s.split("\n")
        info = "%s(%s), " % (self.__class__.__name__, self.approx)
        n = len(info)

        res = " " * n + elem[0]
        res += "\n"
        res += info + elem[1]

        return res

    def __mul__(self, b):
        """

        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2*sqrt_3
        >>> p.poly.printCoeff()
        '[-6,0,1]'

        """
        Q = an_mul(self.poly, b.poly)

        res = AlgebraicNumber(Q, self.approx * b.approx)

        return res

    def __truediv__(self, b):
        """

        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2/sqrt_3
        >>> p.poly.printCoeff()
        '[-2,0,3]'

        """
        ib = b.inverse()
        return self * ib

    def __neg__(self):
        """

        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> p = -sqrt_2
        >>> p.poly.printCoeff()
        '[-2,0,1]'

        """
        mx = QPolynomial(coeff=[0, -1])
        p2 = self.poly.compose(mx)

        res = AlgebraicNumber(p2, -self.approx)

        return res

    def __sub__(self, b):
        nb = -b
        return self + nb

    def __add__(self, b):
        """

        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2+sqrt_3
        >>> p.poly.printCoeff()
        '[1,0,-10,0,1]'
        >>> ref = np.sqrt(2) + np.sqrt(3)
        >>> np.abs(p.approx - ref) < 1e-10
        True

        """
        Q = an_add(self.poly, b.poly)

        res = AlgebraicNumber(Q, self.approx + b.approx)

        return res

    def conj(self):
        """

        >>> z = AlgebraicNumber.unity() + AlgebraicNumber.imaginary()
        >>> z.poly.printCoeff()
        '[2,-2,1]'
        >>> p = z*z.conj()
        >>> p.poly.printCoeff()
        '[-2,1]'

        """
        res = AlgebraicNumber(self.poly, np.conj(self.approx))

        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
