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


def nint(x: np.complex64) -> np.int64:
    """
    
    Returns the nearest integer n from a complex x
    
    n is defined as :
    
    :math:`n = floor ( Re(x)+1/2 )`
    
    Args:
      x
        Input complex
    
    Returns:
      The integer n
      
    """
    if hasattr(x, "__iter__"):
        n = len(x)
        res = np.empty(n, dtype=np.complex64)
        for i in range(n):
            res[i] = nint(x[i])
        return res

    xi = np.int64(np.round(np.real(x), 0))
    yi = np.int64(np.round(np.imag(x), 0))
    # return xi+1j*yi
    return xi


def cpslq(x: np.array) -> np.array:
    """
    PSLQ over Q[sqrt(-D)]
    
    Examples:
      >>> # cpslq(np.array([np.log(2),np.log(3),np.log(4),np.log(6)]))
      
    """
    n = len(x)

    s = np.empty(n)
    for i in range(n):
        s2 = np.sum(np.real(x[i:] * np.conj(x[i:])))
        s[i] = np.sqrt(s2)

    t = s[0]
    y = np.array(x, dtype=np.complex64) / t
    s = s / t

    H = np.zeros((n, n - 1), dtype=np.complex64)
    for i in range(n - 1):
        H[i, i] = s[i + 1] / s[i]
    for i in range(n):
        for j in range(i):
            H[i, j] = -np.conj(y[i]) * y[j] / (s[j] * s[j + 1])

    # print(lin.norm(np.conj(H.T)@H-np.eye(n-1), ord='fro'))
    # print(lin.norm(H, ord='fro'), np.sqrt(n-1))
    # print(lin.norm(x@H=) # = 0

    B = np.eye(n, dtype=np.complex64)
    for i in range(1, n):
        for j in reversed(range(i)):
            t = nint(H[i, j] / H[j, j])
            y[j] = y[j] + t * y[i]
            for k in range(j):
                H[i, k] = H[i, k] - t * H[j, k]
            for k in range(n):
                B[k, j] = B[k, j] + t * B[k, i]

    gam = np.sqrt(4 / 3)
    while True:
        # step 1
        v = [gam ** r * np.abs(H[r, r]) for r in range(n - 1)]
        m = np.argmax(v)

        # step 2
        y[[m, m + 1]] = y[[m + 1, m]]
        H[[m, m + 1], :] = H[[m + 1, m], :]
        B[:, [m, m + 1]] = B[:, [m + 1, m]]

        # step 3
        if m < n - 2:
            # print(H[m:m+2,m:m+2])
            t0 = np.sqrt(
                H[m, m] * np.conj(H[m, m]) + H[m, m + 1] * np.conj(H[m, m + 1])
            )
            t1 = H[m, m] / t0
            t2 = H[m, m + 1] / t0
            for i in range(m, n):
                t3 = H[i, m]
                t4 = H[i, m + 1]
                H[i, m] = np.conj(t1) * t3 + np.conj(t2) * t4
                H[i, m + 1] = -t2 * t3 + t1 * t4
            # print('verif H.H@H', lin.norm(np.conj(H.T)@H-np.eye(n-1), ord='fro'))
            # print(H[m:m+2,m:m+2])

        # step 4
        for i in range(m + 1, n):
            for j in reversed(range(min(i - 1, m + 1))):
                t = nint(H[i, j] / H[j, j])
                y[j] = y[j] + t * y[i]
                for k in range(j):
                    H[i, k] = H[i, k] - t * H[j, k]
                for k in range(n):
                    B[k, j] = B[k, j] + t * B[k, i]

        # step 5
        M = 1 / np.max([lin.norm(H[r, :]) for r in range(n)])

        # step 6
        ny = np.abs(y)
        i0 = np.argmin(ny)

        nH = [np.abs(H[i, i]) for i in range(n - 1)]
        ih0 = np.argmin(nH)

        if ny[i0] < 1e-6 or nH[ih0] < 1e-6:
            vec = B[:, i0].astype("int64")
            print("vec", vec)
            print("y", y)
            print("M", M)
            print("verif", np.sum(vec * x))
            print()
            return vec


def haroldgcd(*args) -> np.array:
    """
    Takes 1D numpy arrays and computes the numerical greatest common
    divisor polynomial. The polynomials are assumed to be in decreasing
    powers, e.g. :math:`s^2 + 5` should be given as ``np.array([1,0,5])``.
    Returns a numpy array holding the polynomial coefficients
    of GCD. The GCD does not cancel scalars but returns only monic roots.
    In other words, the GCD of polynomials :math:`2` and :math:`2s+4` is
    still computed as :math:`1`.
    
    Args:
      args
        A collection of 1D array_likes.
        
    Returns:
      Computed GCD of args.
        
    Examples:
      >>> P1 = [1, 4, 6, 6, 5, 2]
      >>> P2 = [1, 14, 71, 154, 120]
      >>> P3 = [1, 20, 180, 960, 3360, 8064, 13440, 15360, 11520, 5120, 1024]
      >>> a = haroldgcd(P1, P2, P3)
      >>> a
      array([1., 2.]...
      
    .. warning:: It uses the LU factorization of the Sylvester matrix.
                 Use responsibly. It does not check any certificate of
                 success by any means (maybe it will in the future).
                 I have played around with ERES method but probably due
                 to my implementation, couldn't get satisfactory results.
                 I am still interested in better methods.
                 
    """
    raw_arr_args = [np.atleast_1d(np.squeeze(x)) for x in args]
    arr_args = [np.trim_zeros(x, "f") for x in raw_arr_args if x.size > 0]
    dimension_list = [x.ndim for x in arr_args]

    # do we have 2d elements?
    if max(dimension_list) > 1:
        raise ValueError("Input arrays must be 1D arrays, rows, or columns")

    degree_list = np.array([x.size - 1 for x in arr_args])
    max_degree = np.max(degree_list)
    max_degree_index = np.argmax(degree_list)

    try:
        # There are polynomials of lesser degree
        second_max_degree = np.max(degree_list[degree_list < max_degree])
    except ValueError:
        # all degrees are the same
        second_max_degree = max_degree

    n, p, h = max_degree, second_max_degree, len(arr_args) - 1

    # If a single item is passed then return it back
    if h == 0:
        return arr_args[0]

    if n == 0:
        return np.array([1])

    if n > 0 and p == 0:
        return arr_args.pop(max_degree_index)

    # pop out the max degree polynomial and zero pad
    # such that we have n+m columns
    S = np.array(
        [np.hstack((arr_args.pop(max_degree_index), np.zeros((1, p - 1)).squeeze()))]
        * p
    )

    # Shift rows to the left
    for rows in range(S.shape[0]):
        S[rows] = np.roll(S[rows], rows)

    # do the same to the remaining ones inside the regular_args
    for item in arr_args:
        _ = np.array(
            [np.hstack((item, [0] * (n + p - item.size)))] * (n + p - item.size + 1)
        )
        for rows in range(_.shape[0]):
            _[rows] = np.roll(_[rows], rows)
        S = np.r_[S, _]

    rank_of_sylmat = np.linalg.matrix_rank(S)

    if rank_of_sylmat == min(S.shape):
        return np.array([1])
    else:
        p, l, u = lin.lu(S)

    u[abs(u) < 1e-8] = 0
    for rows in range(u.shape[0] - 1, 0, -1):
        if not any(u[rows, :]):
            u = np.delete(u, rows, 0)
        else:
            break

    gcdpoly = np.real(np.trim_zeros(u[-1, :], "f"))
    # make it monic
    gcdpoly /= gcdpoly[0]

    return gcdpoly


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
      >>> polygcd(R, S)
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

    class CheckResult:
        def __str__(self):
            res = []
            for k in self.__dict__.keys():
                res.append("%s=%s" % (k, self.__dict__[k]))
            return ",".join(res)

    res = CheckResult()

    res.ok = 0
    for k, hk in enumerate(h):
        res.k = k
        res.hk = hk

        if np.abs(np.imag(hk)) > tol:
            res.ok = 1
            break

        xk = np.real(hk)
        d = np.abs(xk - np.round(xk, 0))
        if d > tol:
            res.d = d
            res.ok = 2
            break

    return res


def simplify(h, root, tol=1e-9):
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

            res = is_poly_valid(h4, tol)

            if res.ok == 0:
                break

        if res.ok == 0:
            break

    if res.ok != 0:
        raise AssertionError(str(res), h3, h4)

    h5 = np.int32(np.real(h4))

    # Suppressing the first null coefficients
    n = len(h5) - 1

    for i in range(n + 1):
        if h5[i] != 0:
            break

    return h5[i:]


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
