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
    def test_mul(self):
        a = [-2, 0, 1]
        b = [-3, 0, 1]

        coeff = inria_mul(a, b)

        ref = np.array([36, 0, -12, 0, 1])

        self.assertNpArrayAlmostEqual(coeff, ref, delta=1e-9)

    def test_mul2(self):
        a = [-2, 0, 1]
        b = [-2, 0, 1]

        coeff = inria_mul(a, b)

        ref = np.array([16, 0, -8, 0, 1])

        self.assertNpArrayAlmostEqual(coeff, ref, delta=1e-9)

    def test_add(self):
        a = [2, 1]
        b = [-3, 1]

        coeff = inria_add(a, b)

        ref = [-1, 1]

        self.assertNpArrayAlmostEqual(coeff, ref, delta=1e-9)

    def test_add2(self):
        a = [2, 1]
        b = [-2, 1]

        coeff = inria_add(a, b)

        ref = [0, 1]

        self.assertNpArrayAlmostEqual(coeff, ref, delta=1e-9)

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
        
    def test_log_reverse(self):
        a = [-2, 0, 1]
        D = 10
        la = LogarithmicReverse(a, D=D)
        ref = np.zeros(D + 1)
        ref[::2] = 2 ** np.arange(1, 2 + (D + 1) // 2)
        self.assertNpArrayAlmostEqual(la, ref, delta=1e-9)

        h = np.array([1, 2, 3])
        lr = LogarithmicReverse(h)
        ref = np.array([2.0, -2 / 3, -2 / 9])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)

        lr = LogarithmicReverse(h, D=5)
        ref = np.array([2.0, -2 / 3, -2 / 9, 10 / 27, -14 / 81, -2 / 243])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)

        h = np.array([0, -3])
        lr = LogarithmicReverse(h)
        ref = np.array([1])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)

    def test_polynomial_from_log_reverse(self):
        h = np.array([0, -3])
        lr = LogarithmicReverse(h)
        h2 = PolynomialFromLogReverse(lr, D=1)
        self.assertNpArrayAlmostEqual(h, h2 * h[-1], delta=1e-9)

        for _ in range(10):
            n = np.random.randint(low=2, high=9)
            h = [0]
            while h[-1] == 0:
                h = np.random.randint(low=-5, high=5, size=n)

            lr = LogarithmicReverse(h)
            h2 = PolynomialFromLogReverse(lr, D=n - 1)

            self.assertNpArrayAlmostEqual(h, h2 * h[-1], delta=1e-9)


if __name__ == "__main__":
    import unittest

    unittest.main()
