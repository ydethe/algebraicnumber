import collections
from math import gcd, factorial
from functools import reduce
from itertools import combinations

import numpy as np
from numpy.linalg import matrix_rank
from numpy.polynomial import polynomial as P
import scipy.signal as sig
import scipy.linalg as lin


def nCr(n, r):
    return factorial(n) // factorial(r) // factorial(n - r)


def compagnon(h):
    n = len(h) - 1
    res = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        res[i, -1] = -h[i]
        if i <= n - 2:
            res[i + 1, i] = h[-1]

    return res


def kron_mul(a: np.array, b: np.array) -> np.array:
    """
    
    >>> sqrt_2 = [-4,0,2]
    >>> sqrt_3 = [-9,0,3]
    >>> p = kron_mul(sqrt_2, sqrt_3)
    >>> # p.coeff
    # [-6, 0, 1]
    
    """
    Ma = compagnon(a)
    if Ma.shape[0] > 1:
        Ka = Ma[1, 0]
    else:
        Ka = 1

    Mb = compagnon(b)
    if Mb.shape[0] > 1:
        Kb = Mb[1, 0]
    else:
        Kb = 1

    Mc = np.kron(Ma, Mb)

    P = np.poly(Mc)
    P = np.array([int(x) for x in np.round(P, 0)[-1::-1]])
    n = len(P)
    Q = P * (Ka * Kb) ** np.arange(n)

    return Q


def kron_add(a: np.array, b: np.array) -> np.array:
    """
    
    >>> sqrt_2 = [-4,0,2]
    >>> sqrt_3 = [-9,0,3]
    >>> p = kron_add(sqrt_2, sqrt_3)
    >>> # p.coeff
    # [1, 0, -10, 0, 1]
    
    """
    Ma = compagnon(a)
    if Ma.shape[0] > 1:
        Ka = Ma[1, 0]
    else:
        Ka = 1

    Mb = compagnon(b)
    if Mb.shape[0] > 1:
        Kb = Mb[1, 0]
    else:
        Kb = 1

    I = np.eye(Ma.shape[0])
    J = np.eye(Mb.shape[0])

    Mc = np.kron(Ma, J) * Kb + np.kron(I, Mb) * Ka
    Kc = Ka * Kb

    P = np.poly(Mc)
    P = np.array([int(x) for x in np.round(P, 0)[-1::-1]])
    n = len(P)
    Q = P * Kc ** np.arange(n)

    return Q


def sylvester(R, S):
    """
    
    Examples:
      >>> R = [1, -3, 1, -3, 1]
      >>> S = [-3, 2, -9, 4]
      >>> m = sylvester(R, S)
      
    """
    r = P.polytrim(R)
    s = P.polytrim(S)

    m = len(r) - 1
    n = len(s) - 1
    res = np.zeros((n + m, n + m))
    for k in range(n):
        res[k, k : k + m + 1] = r[-1::-1]
    for k in range(m):
        res[k + n, k : k + n + 1] = s[-1::-1]

    return res


def resultant(R, S):
    """
    
    Examples:
      >>> R = [1, -3, 1, -3, 1]
      >>> S = [-3, 2, -9, 4]
      >>> resultant(R, S)
      -4563.0
      
    """
    m = sylvester(R, S)
    return lin.det(m)


def discriminant(R):
    """
    
    Examples:
      >>> R = [1, -3, 1, -3]
      >>> d = discriminant(R)
      >>> d0 = -400
      >>> np.abs(d-d0) < 1e-10
      True
      
    """
    r = P.polytrim(R)
    n = len(r) - 1

    if ((n * (n - 1)) // 2) % 2 == 0:
        s = 1
    else:
        s = -1

    dp = P.polyder(r)
    res = resultant(r, dp)

    dis = s * res / r[-1]

    return dis


def mahler_separation_bound(R: np.array) -> float:
    r"""The minimum root separation for a polynomial P is defined as:
    
    .. math::
        sep(P) = min(|r-s|, (r,s) \in roots(P), r \neq s)
    
    *In the case of a square-free polynomial R of degree d and with integer coefficients*,
    this function gives a lower bound (Mahler, 1964) :
    
    .. math::
        sep(P) > \sqrt{\frac{3.|D|}{d^{d+2}}}. || R ||_2^{1-d}
    
    where D is the discriminant of the polynomial
    
    Examples:
      >>> R = [1, -3, 1, -3]
      >>> b = mahler_separation_bound(R)
      >>> b
      0.111...
      
    """
    r = P.polytrim(R)
    n = len(r) - 1

    d = discriminant(R)
    sep = np.sqrt(3 * np.abs(d) / n ** (n + 2)) / lin.norm(r, ord=2) ** (n - 1)

    return sep


def polygcd(R, S):
    r"""
    
    Examples:
      >>> R = [1, -1, 0, 0, -1, 1]
      >>> S = [-1, 0, 0, -4, 5]
      >>> polygcd(S, R)
      array([-1.,  1.]...
      
    """
    a = P.polytrim(R)
    b = P.polytrim(S)

    m = len(a) - 1
    n = len(b) - 1

    if n > m:
        a, b = b, a
        n, m = m, n

    # Here, deg(a) >= deb(b)

    while True:
        q, r = P.polydiv(a, b)

        a, b = b, r

        if len(b) == 1 and b[0] == 0:
            return a / a[-1]


def square_free_fact(R):
    r"""
    
    Examples:
      >>> R = [1, -1, 0, 0, -1, 1]
      >>> square_free_fact(R)
      array([-1,  0,  0,  0,  1]...
      
    """
    p = P.polytrim(R)
    n = len(p) - 1

    while True:
        dp = P.polyder(p)
        g = polygcd(p, dp)
        if len(g) == 1 and g[0] == 1:
            return np.int32(np.round(p, 0))

        p, r = P.polydiv(p, g)
        if len(r) != 1 or r[0] != 0:
            raise AssertionError(r)


def polytrans(R, a):
    r"""
    
    Examples:
      >>> R = [0, 0, 1]
      >>> polytrans(R, 2)
      array([4., 4., 1.]...
      
    """
    r = P.polytrim(R)
    n = len(r) - 1

    q = np.empty(n + 1)
    for k in range(n + 1):
        q[k] = 0
        for p in range(k, n + 1):
            q[k] += nCr(p, k) * r[p] * a ** (p - k)

    return q


def polycomp(R, S):
    r"""Computes R(S(X))
    
    Examples:
      >>> R = [1, -1, 1]
      >>> S = [-4, 5]
      >>> polycomp(R, S)
      array([ 21., -45.,  25.]...
      
    """
    r = P.polytrim(R)
    s = P.polytrim(S)

    res = [0]
    sp = [1]
    for rk in r:
        if rk != 0:
            res = P.polyadd(res, P.polymul([rk],sp))
        sp = P.polymul(sp, s)
        
    return res
    
    
def npolymul(*polynomials):
    res = [1]
    for q in polynomials:
        res = P.polymul(res, q)
    return res


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


def simplify(h, root, tol=1e-7):
    r"""
    
    Examples:
      >>> R = [4, -2, 0, -2, 1]
      >>> simplify(R, 2)
      array([-2,  1]...
      >>> simplify(R, 2**(1/3))
      array([-2,  0,  0,  1]...
      
    """
    # 1. square free
    h2 = square_free_fact(h)

    g = reduce(gcd, h2)
    h3 = h2 // g

    n = len(h3) - 1
    roots = P.polyroots(h3)

    # We remove the given root
    yy = np.abs(roots - root)
    i0 = np.argmin(yy)
    roots = np.delete(roots, i0)

    if n <= 1:
        return h3

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

    return h6


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
    