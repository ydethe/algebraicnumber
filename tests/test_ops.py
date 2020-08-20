from math import factorial
from unittest.mock import Mock

import numpy as np
from numpy.polynomial import polynomial as P

from AlgebraicNumber.AlgebraicNumber import AlgebraicNumber
from AlgebraicNumber.inria_utils import *

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestOperations(TestBase):
    def test_add3(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber([-2, 0, 1], -np.sqrt(2))

        z = a + b

        self.assertQPolynomialEqual(z, AlgebraicNumber.zero())

    def test_sub(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber([-2, 0, 1], -np.sqrt(2))

        z = a - b

        ref = AlgebraicNumber([-8, 0, 1], np.sqrt(8))
        self.assertQPolynomialEqual(z, ref)

    def test_div(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber([-2, 0, 1], -np.sqrt(2))

        z = a / b

        ref = AlgebraicNumber([1, 1], -1)

        self.assertQPolynomialEqual(z, ref)

    def test_div2(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber.zero()

        def test_div(a, b):
            return a / b

        self.assertRaises(ZeroDivisionError, test_div, a, b)

    def test_cycle(self):
        n = 3
        a = AlgebraicNumber([-1] + [0] * (n - 1) + [1], np.exp(1j * 2 * np.pi / n))

        ref = AlgebraicNumber(n*[1], np.exp(1j * 2 * np.pi / n))
        self.assertQPolynomialEqual(a, ref)

        z = AlgebraicNumber.unity()

        z2 = AlgebraicNumber.integer(2)
        ref = AlgebraicNumber([-2, 1], 2)
        self.assertQPolynomialEqual(z2, ref)

        for i in range(n):
            print('Cycle %i/%i' % (i,n))
            z = z * a

        ref = AlgebraicNumber([-1, 1], 1)
        self.assertQPolynomialEqual(z, ref)

    def test_plot_roots(self):
        n = 5
        a = AlgebraicNumber([-1] + [0] * (n - 1) + [1], np.exp(1j * 2 * np.pi / n))

        axe = Mock()
        a.plotRoots(axe)

    def ntest_ramanujan(self):
        # a = AlgebraicNumber([-1, 0, 0, 9], 9**(-1/3))
        # b = AlgebraicNumber([-2, 0, 0, 9], (2/9)**(1/3))
        # c = AlgebraicNumber([-4, 0, 0, 9], (4/9)**(1/3))
        # z = a-b+c

        d = AlgebraicNumber([-2, 0, 0, 1], 2 ** (1 / 3))
        e = d - AlgebraicNumber.unity()
        z2 = e.pow(1, 3)

        ref = np.array([-3, 0, 0, 3, 0, 0, -3, 0, 0, 1])
        self.assertNpArrayAlmostEqual(z2.coeff, ref, delta=1e-9)

        # Egalite de Ramanujan
        # self.assertNpArrayAlmostEqual(z.coeff, z2.coeff, delta=1e-9)


if __name__ == "__main__":
    import unittest

    unittest.main()
