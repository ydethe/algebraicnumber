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

        ref = np.array([0, 1])

        self.assertNpArrayAlmostEqual(z.coeff, ref, delta=1e-9)

    def test_sub(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber([-2, 0, 1], -np.sqrt(2))

        z = a - b

        ref = np.array([-8, 0, 1])
        self.assertNpArrayAlmostEqual(z.coeff, ref, delta=1e-9)

    def test_div(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber([-2, 0, 1], -np.sqrt(2))

        z = a / b

        ref = np.array([1, 1])

        self.assertNpArrayAlmostEqual(z.coeff, ref, delta=1e-9)

    def test_div2(self):
        a = AlgebraicNumber([-2, 0, 1], np.sqrt(2))
        b = AlgebraicNumber.zero()

        def test_div(a, b):
            return a / b

        self.assertRaises(ZeroDivisionError, test_div, a, b)

    def test_cycle(self):
        n = 5
        a = AlgebraicNumber([-1] + [0] * (n - 1) + [1], np.exp(1j * 2 * np.pi / n))

        ref = np.array([1, 1, 1, 1, 1])
        self.assertNpArrayAlmostEqual(a.coeff, ref, delta=1e-9)

        z = AlgebraicNumber.unity()

        z2 = AlgebraicNumber.integer(2)
        ref = np.array([-2, 1])
        self.assertNpArrayAlmostEqual(z2.coeff, ref, delta=1e-9)

        for i in range(n):
            z = z * a

        ref = np.array([-1, 1])
        self.assertNpArrayAlmostEqual(z.coeff, ref, delta=1e-9)

    def test_plot_roots(self):
        n = 5
        a = AlgebraicNumber([-1] + [0] * (n - 1) + [1], np.exp(1j * 2 * np.pi / n))

        axe = Mock()
        a.plotRoots(axe)

    def test_ramanujan(self):
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
