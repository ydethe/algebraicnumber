# utils.py

from math import factorial
from fractions import Fraction
from typing import Iterable

import numpy as np
from scipy import linalg as lin

from AlgebraicNumber.QPolynomial import QPolynomial


def inria_add(a: QPolynomial, b: QPolynomial) -> QPolynomial:
    Da = a.deg()
    Db = b.deg()
    D = Da * Db

    Ea = QPolynomial(q_coeff=[factorial(n) for n in range(Da + 1)])
    Eb = QPolynomial(q_coeff=[factorial(n) for n in range(Db + 1)])
    E2 = QPolynomial(q_coeff=[factorial(n) for n in range(2 * D + 1)])

    la = a.LogRev(D=D)
    lb = b.LogRev(D=D)

    lea = la.termWiseMul(Ea)
    leb = lb.termWiseMul(Eb)
    q = lea * leb
    lp = q.termWiseDiv(E2)
    # lp = P.polymul(la * Ea, lb * Eb) / E2
    # lp = lp[: D + 1]

    res = lp.InvertLogRev()

    return res


def inria_mul(a: QPolynomial, b: QPolynomial) -> QPolynomial:
    Da = a.deg()
    Db = b.deg()
    D = Da + Db

    la = a.LogRev(D=D)
    lb = b.LogRev(D=D)

    lp = la.termWiseMul(lb)

    res = lp.InvertLogRev()

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
