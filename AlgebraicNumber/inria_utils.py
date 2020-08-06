# utils.py

from math import factorial

import numpy as np
from numpy.polynomial import polynomial as P
from scipy import linalg as lin


def newton_sum(h: np.array, d: int) -> float:
    """Computes the d-th Newton's sum :math:`s_d` of the polynomial h
    
    Given the roots :math:`x_k` of h :
    
    :math:`h(x_k) = 0`
    
    .. math::
        s_d=\sum_{k=1}^{n} x_k^d
    
    Args:
      h
        Input polynomial
      d
        Order of the Newton's sum
        
    Returns:
      The d-th Newton's sum of h's roots
      
    Examples:
      >>> h = np.array([1, 2, 3])
      >>> for d in range(5): print(newton_sum(h, d))
      2
      -0.6666666666...
      -0.2222222222...
      0.37037037037...
      -0.1728395061...
      
    """
    # Dh is the degree of h
    h = P.polytrim(h)
    Dh = len(h) - 1

    if d == 0:
        return Dh

    # X is the vector of the d-th Newton's sum s_d :
    # with x_k a root of h,
    # s_d = x_1^d + x_2^d + ... + x_n^d
    # X = [s_1, s_2, ..., s_D]
    A = np.zeros((Dh, Dh))
    B = np.empty(Dh)

    for r in range(Dh):
        A[r, : r + 1] = h[-1 - r :]
        B[r] = -(r + 1) * h[-r - 2]
    X = lin.inv(A) @ B
    r = np.abs(lin.det(A))

    if d <= Dh:
        return X[d - 1]

    a_lrs = -h[:-1] / h[-1]
    for d in range(Dh + 1, d + 1):
        sd = np.sum(a_lrs * X)
        X = np.hstack((X[1:], [sd]))

    return sd


def PolynomialReverse(h: np.array, D: int = None) -> np.array:
    """This function returns the reverse polynomial associated with h, denoted rev(h)

    h is a polynomial with integer coefficients :

    :math:`a_0 + a_1.X + a_2.X^2 + ...`

    h is represented by the sequence [a0, a1, a2, ...]

    rev(h) is defined as follows:

    :math:`rev(h) = X^{deg(h)}.h(1/X)`
    
    If D is > deg(h), the reversed h will be multiplied by :math:`X^k` so that the final degree is D
    
    Args:
      h
        The input polynomial
      D
        The degree of the output. Must be >= deg(h)
        
    Returns:
      The reverse

    Examples:
      >>> h = np.array([1, 2, 3])
      >>> PolynomialReverse(h)
      array([3, 2, 1]...

    """
    if D is None:
        D = len(h) - 1
    rev = np.pad(h[-1::-1], (D + 1 - len(h), 0), "constant", constant_values=(0, 0))
    return rev


def LogarithmicReverse(h: np.array, D: int = None, trim_res: bool = True) -> np.array:
    """This function returns the logarithmic reverse rational power series associated with h, denoted LogRev(h)
    The result of this function is a truncature of degree D
    
    h is a polynomial with integer coefficients :
    
    :math:`a_0 + a_1.X + a_2.X^2 + ...`
    
    h is represented by the sequence [a0, a1, a2, ...]
    
    LogRev(h) is defined as follows :
    
    :math:`LogRev(h) = rev(h')/rev(h)`
    
    Args:
      h
        The input polynomial
      D
        The degree of the output polynomial. By default, same degree as h
      trim_res
        True if you want the output array to be trimmed
        
    Returns:
      The logarithmic reverse

    Examples:
      >>> h = np.array([1, 2, 3])
      >>> LogarithmicReverse(h)
      array([ 2.        , -0.66666667, -0.22222222]...
      >>> lr = LogarithmicReverse(h, D=5)
      >>> lr[3:]
      array([ 0.37037037, -0.17283951, -0.00823045]...
      
    """
    # Dh is the degree of the resulting h
    h = P.polytrim(h)
    Dh = len(h) - 1

    if D is None:
        D = Dh

    lr = np.empty(D + 1)
    for d in range(D + 1):
        lr[d] = newton_sum(h, d)

    if trim_res:
        res = P.polytrim(lr)
    else:
        res = lr

    return res


def PolynomialFromLogReverse(lr: np.array, D: int = None) -> np.array:
    """Given a polynomial lr which is LogRev of h, finds back h
    
    Args:
      lr
        The input polynomial
      D
        The degree of the original polynomial h.
        By default, lr must have its 0 degree coefficient equal to the original polynomial's degree
        
    Returns:
      The original polynomial
      
    Examples:
      >>> h = np.array([2, 3, 4])
      >>> lr = LogarithmicReverse(h)
      >>> PolynomialFromLogReverse(lr)
      array([0.5 , 0.75, 1.  ]...
      
      >>> h = np.array([-1, 0, 1])
      >>> lr = LogarithmicReverse(h, D=4)
      >>> lr
      array([2., 0., 2., 0., 2.]...
      >>> PolynomialFromLogReverse(lr, D=2)
      array([-1.,  0.,  1.]...
      
    """
    # D is the degree of the resulting h
    if D is None:
        D = np.int64(lr[0])

    # Computation of D - LogRev(h) as the fraction n1, d1
    dif = P.polysub([D], lr)

    if dif[0] != 0:
        raise AssertionError(dif)

    # D - LogRev is always dividable by X
    # n2 is the resulting fraction : 1/X.(D - LogRev(h))
    n2 = dif[1:]

    # Integrate n2
    if len(n2) == 0:
        n3 = []
    else:
        n3 = P.polyint(n2)

    if D + 1 > len(n3):
        n3 = np.pad(n3, (0, D + 1 - len(n3)), "constant", constant_values=(0, 0))
    elif D + 1 < len(n3):
        n3 = n3[: D + 1]

    # Exponentiate n3
    n4 = np.zeros(D + 1)
    dl = np.zeros(D + 1)
    n4[0] = 1.0
    k = np.arange(D + 1, dtype=np.int64)
    coeff = np.array([factorial(x) for x in k])
    for i in range(1, D + 1):
        # dl : DL de exp(n3[i].x^i) a l'ordre D
        s = slice(0, D + 1, i)
        nc = 1 + D // i
        dl[:] = 0
        dl[s] = n3[i] ** k[:nc] / coeff[:nc]
        n4 = P.polymul(n4, dl)

    return PolynomialReverse(n4[: D + 1], D=D)


def inria_add(a: np.array, b: np.array) -> np.array:
    Da = len(a) - 1
    Db = len(b) - 1
    D = Da * Db

    Ea = np.array([1 / factorial(n) for n in range(Da + 1)])
    Eb = np.array([1 / factorial(n) for n in range(Db + 1)])
    E2 = np.array([1 / factorial(n) for n in range(2 * D + 1)])

    la = LogarithmicReverse(a, D=D, trim_res=False)
    lb = LogarithmicReverse(b, D=D, trim_res=False)

    lp = P.polymul(la * Ea, lb * Eb) / E2
    lp = lp[: D + 1]

    coeff = PolynomialFromLogReverse(lp, D=Da * Db)

    return coeff


def inria_mul(a: np.array, b: np.array) -> np.array:
    Da = len(a) - 1
    Db = len(b) - 1
    D = Da + Db

    la = LogarithmicReverse(a, D=D, trim_res=False)
    lb = LogarithmicReverse(b, D=D, trim_res=False)

    lp = la * lb

    coeff = PolynomialFromLogReverse(lp, D=D)

    return coeff


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
