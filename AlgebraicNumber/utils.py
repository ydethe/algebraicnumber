import collections
from math import gcd, factorial
from functools import reduce
from itertools import combinations

import numpy as np
from numpy.polynomial import polynomial as P
from numpy.linalg import matrix_rank
import scipy.signal as sig
import scipy.linalg as lin


def nCr(n, r):
    return factorial(n) // factorial(r) // factorial(n - r)


def prod(*iterable, start=1):
    for i in iterable:
        start = start * i
    return start


def is_poly_valid(h, tol):
    """The coeficients of the polynomial h must:
    * have no imaginary part
    * be integers
    """

    ok = 0
    for k, hk in enumerate(h):
        if np.abs(np.imag(hk)) > tol:
            ok = 1
            break

        xk = np.real(hk)
        d = np.abs(xk - np.round(xk, 0))
        if d > tol:
            ok = 2
            break

    return ok


def npolymul(*polynomials):
    res = [1]
    for q in polynomials:
        res = P.polymul(res, q)
    return res


def simplify(h: "QPolynomial", root: np.complex64, tol: float = 1e-7) -> "QPolynomial":
    r"""
    
    Examples:
      >>> from AlgebraicNumber.QPolynomial import QPolynomial
      >>> R = QPolynomial(p_coeff=[4, -2, 0, -2, 1])
      >>> simplify(R, 2)
      [-2  1]
      [1 1]
      >>> simplify(R, 2**(1/3))
      [-2  0  0  1]
      [1 1 1 1]
      
    """
    from AlgebraicNumber.QPolynomial import QPolynomial

    # 1. square free
    h1 = h.squareFreeFact().getCoefficientsAsFraction()
    lq = np.array([x.denominator for x in h1], dtype=np.int64)
    ppcm = reduce(lambda x, y: (x * y) // gcd(x, y), lq)

    h2 = np.array([x.numerator for x in h1], dtype=np.int64)
    h2 *= ppcm // lq

    g = reduce(gcd, h2)
    h3 = h2 // g

    n = len(h3) - 1
    roots = P.polyroots(h3)

    # We remove the given root
    yy = np.abs(roots - root)
    i0 = np.argmin(yy)
    roots = np.delete(roots, i0)

    if n <= 1:
        return QPolynomial(p_coeff=h3)

    for nr in range(n):
        for c in combinations(roots, nr):
            h4 = npolymul(*[[-x, 1] for x in c])
            h4 = P.polymul(h4, [-root * h3[-1], h3[-1]])

            ok = is_poly_valid(h4, tol)

            if ok == 0:
                break

        if ok == 0:
            break

    if ok != 0:
        raise AssertionError(ok, h3, h4)

    h5 = np.int32(np.round(np.real(h4), 0))

    # Suppressing the first null coefficients
    n = len(h5) - 1

    for i in range(n + 1):
        if h5[i] != 0:
            break

    h6 = h5[i:]
    if len(h6) == 1:
        h6 = np.hstack(([0], h6))

    return QPolynomial(p_coeff=h6)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
