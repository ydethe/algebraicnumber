# utils.py

from math import factorial

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import linalg as lin


def PolynomialReverse(h):
    '''

    h is a polynomial with integer coefficients :

    :math:`a_0 + a_1.X + a_2.X^2 + ...`

    h is represented by the sequence [a0, a1, a2, ...]

    This function returns the reverse polynomial associated with h, denoted rev(h) :

    :math:`rev(h) = X^{deg(h)}.h(1/X)`

    Args:
      h
        The input polynomial

    Returns:
      The reverse

    Examples:
      >>> h = np.array([1, 2, 3])
      >>> PolynomialReverse(h)
      array([3, 2, 1])

    '''
    return h[-1::-1]


def LogarithmicReverse(h):
    '''

    h is a polynomial with integer coefficients :

    :math:`a_0 + a_1.X + a_2.X^2 + ...`

    h is represented by the sequence [a0, a1, a2, ...]

    This function returns the logarithmic reverse rational power series associated with h, denoted LogRev(h) :

    :math:`LogRev(h) = rev(h')/rev(h)`

    Args:
      h
        The input polynomial

    Returns:
      The logarithmic reverse

    Examples:
      >>> h = np.array([1, 2, 3])
      >>> LogarithmicReverse(h)
      array([ 2.        , -0.66666667, -0.22222222])

    '''
    # D is the degree of the resulting h
    D = len(h)-1

    # X is the vector of the d-th Newton's sum s_d :
    # with x_k a root of h, 
    # s_d = x_1^d + x_2^d + ... + x_n^d
    # X = [s_1, s_2, ..., s_D]
    A = np.zeros((D,D))
    B = np.empty(D)
    
    for r in range(D):
        A[r,:r+1] = h[-1-r:]
        B[r] = -(r+1)*h[-r-2]
    X = lin.inv(A)@B
    
    lr = np.hstack(([D], X))
    return lr


def PolynomialFromLogReverse(lr):
    '''

    Examples:
      >>> h = np.array([2, 3, 4])
      >>> lr = LogarithmicReverse(h)
      >>> PolynomialFromLogReverse(lr)
      array([0.5 , 0.75, 1.  ])
      
    '''
    # D is the degree of the resulting h
    D = len(lr)-1

    # Computation of D - LogRev(h) as the fraction n1, d1
    dif = poly.polysub([D], lr)
    assert(dif[0] == 0)

    # D - LogRev is always dividable by X
    # n2 is the resulting fraction : 1/X.(D - LogRev(h))
    n2 = dif[1:]

    # Integrate n2
    n3 = poly.polyint(n2)
    n3 = np.pad(n3, (0, D+1-len(n3)), 'constant', constant_values=(0,0))
    # print('n3', n3)

    # Exponentiate n3
    n4 = np.zeros(D+1)
    dl = np.zeros(D+1)
    n4[0] = 1.
    k = np.arange(D+1, dtype=np.int64)
    coeff = np.array([factorial(x) for x in k])
    for i in range(1,D+1):
        # dl : DL de exp(n3[i].x^i) a l'ordre D
        s = slice(0,D+1,i)
        nc = 1+D//i
        dl *= 0
        dl[s] = n3[i]**k[:nc]/coeff[:nc]
        n4 = poly.polymul(n4,dl)
        
    return PolynomialReverse(n4[:D+1])


if __name__ == '__main__':
    import doctest
    doctest.testmod()