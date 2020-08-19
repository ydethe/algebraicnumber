from math import factorial
from unittest.mock import Mock

import numpy as np
from numpy.polynomial import polynomial as P

from AlgebraicNumber.AlgebraicNumber import AlgebraicNumber
from AlgebraicNumber.inria_utils import *
from AlgebraicNumber.QPolynomial import QPolynomial

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestINRIA(TestBase):
    def test_mul(self):
        a = QPolynomial(coeff=[-2, 0, 1])
        b = QPolynomial(coeff=[-3, 0, 1])

        coeff = inria_mul(a, b)

        ref = QPolynomial(coeff=[36, 0, -12, 0, 1])

        self.assertQPolynomialEqual(coeff, ref)

    def test_mul2(self):
        a = QPolynomial(coeff=[-2, 0, 1])
        b = QPolynomial(coeff=[-2, 0, 1])

        coeff = inria_mul(a, b)

        ref = QPolynomial(coeff=[16, 0, -8, 0, 1])

        self.assertQPolynomialEqual(coeff, ref)

    def test_add(self):
        a = QPolynomial(coeff=[2, 1])
        b = QPolynomial(coeff=[-3, 1])

        coeff = inria_add(a, b)

        ref = QPolynomial(coeff=[-1, 1])

        self.assertQPolynomialEqual(coeff, ref)

    def test_add2(self):
        a = QPolynomial(coeff=[2, 1])
        b = QPolynomial(coeff=[-2, 1])

        coeff = inria_add(a, b)

        ref = QPolynomial(coeff=[0, 1])

        self.assertQPolynomialEqual(coeff, ref)

    def test_polynomial_from_log_reverse(self):
        for _ in range(10):
            n = np.random.randint(low=2, high=9)
            p = [0]
            while p[-1] == 0:
                p = np.random.randint(low=-5, high=5, size=n)
            q = np.random.randint(low=1, high=5, size=n)

            h = QPolynomial(coeff=[Fraction(int(x), int(y)) for (x, y) in zip(p, q)])
            lr = h.LogRev()
            h2 = lr.InvertLogRev()

            self.assertQPolynomialEqual(h, h2 * h[-1])


if __name__ == "__main__":
    import unittest

    unittest.main()
