"""
Implements the PSLQ algorithm for integer relation detection,
and derivative algorithms for constant recognition.
"""

import numpy as np
import scipy.linalg as lin


def nint(x):
    xi = np.int64(np.round(x,0))
    
    return xi

def acot(x):
    return np.arctan(1/x)
    
def pslq(x, tol=1e-10, maxcoeff=1000, maxsteps=100, verbose=False):
    r"""
    Given a vector of real numbers `x = [x_0, x_1, ..., x_n]`, ``pslq(x)``
    uses the PSLQ algorithm to find a list of integers
    `[c_0, c_1, ..., c_n]` such that

    .. math ::

        |c_1 x_1 + c_2 x_2 + ... + c_n x_n| < \mathrm{tol}

    and such that `\max |c_k| < \mathrm{maxcoeff}`. If no such vector
    exists, :func:`~identification.pslq` returns ``None``. The tolerance defaults to
    3/4 of the working precision.

    **Examples**

    Find rational approximations for `\pi`::

        >>> pslq([-1, np.pi], tol=0.01)
        array([22,  7])
        >>> pslq([-1, np.pi], tol=0.001)
        array([355, 113])

    Pi is not a rational number with denominator less than 1000::

        >>> pslq([-1, np.pi])
        

    To within the standard precision, it can however be approximated
    by at least one rational number with denominator less than `10^{12}`::

        >>> p, q = pslq([-1, np.pi], maxcoeff=10**12)
        >>> print(p); print(q)
        7049532881
        2243936009
        
    The PSLQ algorithm can be applied to long vectors. For example,
    we can investigate the rational (in)dependence of integer square
    roots::

        >>> pslq([np.sqrt(n) for n in range(2, 5+1)])
        >>>
        >>> pslq([np.sqrt(n) for n in range(2, 8+1)])
        array([ 2,  0,  0,  0,  0,  0, -1])

    **Machin formulas**

    A famous formula for `\pi` is Machin's,

    .. math ::

        \frac{\pi}{4} = 4 \operatorname{acot} 5 - \operatorname{acot} 239

    There are actually infinitely many formulas of this type. Two
    others are

    .. math ::

        \frac{\pi}{4} = \operatorname{acot} 1

        \frac{\pi}{4} = 12 \operatorname{acot} 49 + 32 \operatorname{acot} 57
            + 5 \operatorname{acot} 239 + 12 \operatorname{acot} 110443

    We can easily verify the formulas using the PSLQ algorithm::

        >>> pslq([np.pi/4, acot(1)])
        array([ 1, -1])
        >>> pslq([np.pi/4, acot(5), acot(239)])
        array([ 1, -4,  1])
        >>> pslq([np.pi/4, acot(49), acot(57), acot(239), acot(110443)])
        array([  1, -12, -32,   5, -12])

    We could try to generate a custom Machin-like formula by running
    the PSLQ algorithm with a few inverse cotangent values, for example
    acot(2), acot(3) ... acot(10). Unfortunately, there is a linear
    dependence among these values, resulting in only that dependence
    being detected, with a zero coefficient for `\pi`::

        >>> pslq([np.pi] + [acot(n) for n in range(2,11)])
        array([ 0,  1, -1,  0,  0,  0, -1,  0,  0,  0])

    We get better luck by removing linearly dependent terms::

        >>> pslq([np.pi] + [acot(n) for n in range(2,11) if n not in (3, 5)])
        array([ 1, -8,  0,  0,  4,  0,  0,  0])

    In other words, we found the following formula::

        >>> 8*acot(2) - 4*acot(7)
        3.141592653589793...
        >>> np.pi
        3.141592653589793...

    **Algorithm**

    This is a fairly direct translation to Python of the pseudocode given by
    David Bailey, "The PSLQ Integer Relation Algorithm":
    http://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html

    The present implementation uses fixed-point instead of floating-point
    arithmetic, since this is significantly (about 7x) faster.
    """

    n = len(x)
    if n < 2:
        raise ValueError("n cannot be less than 2")

    # Convert to fixed-point numbers. The dummy None is added so we can
    # use 1-based indexing. (This just allows us to be consistent with
    # Bailey's indexing. The algorithm is 100 lines long, so debugging
    # a single wrong index can be painful.)
    x = np.hstack(([np.nan], np.array(x, dtype=np.float64)))

    # Sanity check on magnitudes
    minx = min(abs(xx) for xx in x[1:])
    if minx < tol/100:
        if verbose:
            print("STOPPING: (one number is too small)")
        return None

    g = np.sqrt(4/3)
    B = np.eye(n+1, dtype=np.int64)
    H = np.zeros((n+1,n))
    # Initialization
    # step 2
    s = [None] + [0] * n
    for k in range(1, n+1):
        t = 0
        for j in range(k, n+1):
            t += x[j]**2
        s[k] = np.sqrt(t)
    t = s[1]
    y = x[:]
    for k in range(1, n+1):
        y[k] = x[k]/t
        s[k] = s[k]/t
    # step 3
    for i in range(1, n+1):
        if i <= n-1:
            if s[i]:
                H[i,i] = s[i+1] / s[i]
        for j in range(1, i):
            sjj1 = s[j]*s[j+1]
            if sjj1:
                H[i,j] = -y[i]*y[j]/sjj1
    # step 4
    for i in range(2, n+1):
        for j in range(i-1, 0, -1):
            #t = np.floor(H[i,j]/H[j,j] + 0.5)
            if H[j,j]:
                t = nint(H[i,j]/H[j,j])
            else:
                #t = 0
                continue
            y[j] = y[j] + t*y[i]
            for k in range(1, j+1):
                H[i,k] = H[i,k] - t*H[j,k]
            for k in range(1, n+1):
                B[k,j] = B[k,j] + t*B[k,i]
                
    # Main algorithm
    for REP in range(maxsteps):
        # Step 1
        m = -1
        szmax = -1
        for i in range(1, n):
            h = H[i,i]
            sz = g**i * abs(h)
            if sz > szmax:
                m = i
                szmax = sz
        # Step 2
        y[m], y[m+1] = y[m+1], y[m]
        tmp = {}
        for i in range(1,n): H[m,i], H[m+1,i] = H[m+1,i], H[m,i]
        for i in range(1,n+1): B[i,m], B[i,m+1] = B[i,m+1], B[i,m]
        # Step 3
        if m <= n - 2:
            t0 = np.sqrt(H[m,m]**2 + H[m,m+1]**2)
            # A zero element probably indicates that the precision has
            # been exhausted. XXX: this could be spurious, due to
            # using fixed-point arithmetic
            if not t0:
                break
            t1 = H[m,m]/t0
            t2 = H[m,m+1]/t0
            for i in range(m, n+1):
                t3 = H[i,m]
                t4 = H[i,m+1]
                H[i,m] = t1*t3+t2*t4
                H[i,m+1] = -t2*t3+t1*t4
                
        # Step 4
        for i in range(m+1, n+1):
            for j in range(min(i-1, m+1), 0, -1):
                try:
                    t = nint(H[i,j]/H[j,j])
                # Precision probably exhausted
                except ZeroDivisionError:
                    break
                y[j] = y[j] + t*y[i]
                for k in range(1, j+1):
                    H[i,k] = H[i,k] - t*H[j,k]
                for k in range(1, n+1):
                    B[k,j] = B[k,j] + t*B[k,i]
        # Until a relation is found, the error typically decreases
        # slowly (e.g. a factor 1-10) with each step TODO: we could
        # compare err from two successive iterations. If there is a
        # large drop (several orders of magnitude), that indicates a
        # "high quality" relation was detected. Reporting this to
        # the user somehow might be useful.
        best_err = maxcoeff
        for i in range(1, n+1):
            err = abs(y[i])
            # Maybe we are done?
            if err < tol:
                # We are done if the coefficients are acceptable
                vec = [np.round(B[j,i], 0) for j in range(1,n+1)]
                if max(abs(v) for v in vec) < maxcoeff:
                    if verbose:
                        print("FOUND relation at iter %i/%i" % (REP, maxsteps))
                    return np.array(vec, dtype=np.int64)
            best_err = min(err, best_err)
        # Calculate a lower bound for the norm. We could do this
        # more exactly (using the Euclidean norm) but there is probably
        # no practical benefit.
        recnorm = lin.norm(H, ord=np.inf)
        if recnorm:
            norm = 1 / recnorm
            norm /= 100
        else:
            norm = maxcoeff+1
        if verbose:
            print("%i/%i:  Norm: %s, Err: %s" % (REP, maxsteps, norm, np.min(np.abs(y[1:]))))
        if norm >= maxcoeff:
            break
    if verbose:
        print("CANCELLING after step %i/%i." % (REP, maxsteps))
        print("Could not find an integer relation. Norm bound: %s" % norm)
    return None
    

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    
    