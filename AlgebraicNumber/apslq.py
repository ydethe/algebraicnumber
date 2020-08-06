import numpy as np
import scipy.linalg as lin


def apslq(
    xx: np.array,
    iterations: int = 1000,
    digits: int = 100,
    gmma: float = 2 / np.sqrt(3),
    threshold: float = 10 ** (-12),
    D: int = 0,
) -> np.array:
    """
    APSLQ over Q[sqrt(-D)]
    
    Examples:
      >>> # apslq([np.log(2),np.log(3),np.log(4),np.log(6)], D=0)
      
    """
    # Perform module initialisation for an instance of Q(sqrt(-D)).
    #  ... this sets up the sqrtD, nSqrtD, omega and nOmega module variables.
    sqrtD = np.sqrt(np.abs(D))
    prev_m = 1

    if (D == 0) or (D == 1) or (D % 4 == 2) or (D % 4 == 3):
        if D < 0:
            omega = sqrtD * 1j
        else:
            omega = sqrtD
    elif D % 4 == 1:
        if D < 0:
            sd = sqrtD * 1j
        else:
            sd = sqrtD
        omega = (1 + sd) / 2
    else:
        raise ValueError("D mod 4 = 0")

    def AlgNearest(x, D):
        alpha = np.real(x)
        beta = np.imag(x)

        if -D == 0 or -D == 1:
            a = np.round(alpha, 0)
            b = 0
        elif -D % 4 == 1:
            b = floor(2 * beta / sqrtD)
            a = alpha - 0.5 * b

            a = [np.round(a, 0), np.round(a - 0.5, 0)]

            candidates = np.array(
                [a[0] + b * omega, a[1] + (b + 1) * omega], dtype=np.complex64
            )

            # Calculate square distance (we only need to find the minimum, so the square is fine, and less computationally intensive.
            distance = [z * np.conj(z) for z in candidates - x]

            if distance[0] <= distance[1]:
                a = a[1]
            else:
                a, b = a[2], b + 1

        else:
            a = np.round(alpha, 0)
            b = np.round(beta / sqrtD, 0)

        return a + b * omega

    # Normalize the input.
    yy = np.array(xx, dtype=np.complex64) / lin.norm(xx)

    # Set the number size of the APSLQ instance (the number of input values).
    n = len(yy)

    # Pre compute the array of gamma^k for 1 ≤ k ≤ n-1
    gg = np.array([gmma ** k for k in range(1, n)])

    # PSLQ Initialization

    # The B matrix is an algebraic integer valued matrices with the property that for each column vector, col, of B:
    #   col[1]*xx[1] + ... + col[n]*xx[n] = yy[n].
    B = np.eye(n, dtype=np.complex64)

    # Check for trivial identity.
    pos = np.argmin(np.abs(yy))
    miny = np.abs(yy[pos])
    if (
        miny < threshold
    ):  # No need to normalize because B is the identity matrix, so miny / ||Col(B,pos)|| = miny.
        return B[:, pos]

    s = np.empty(n)
    for k in range(n):
        s2 = np.real(np.sum(yy[k:] * np.conj(yy[k:])))
        s[k] = np.sqrt(s2)

    H = np.empty((n, n - 1), dtype=np.complex64)
    for j in range(n - 1):
        for i in range(j - 1):
            H[i, j] = 0

        H[j, j] = s[j + 1] / s[j]

        for i in range(j + 1, n):
            H[i, j] = -np.conj(yy[i]) * yy[j] / s[j] / s[j + 1]

    # Reduce H
    for i in range(1, n):
        for j in reversed(range(i)):
            tf = AlgNearest(H[i, j] / H[j, j], D)

            yy[j] = yy[j] + tf * yy[i]

            for k in range(j):
                H[i, k] = H[i, k] - tf * H[j, k]

            for k in range(n):
                B[k, j] = B[k, j] + tf * B[k, i]

    for ii in range(iterations):
        # ITERATION
        m, t = 0, 0.0

        for i in range(n - 1):
            if i == prev_m:
                continue

            if t < gg[i] * np.abs(H[i, i]):
                t = gg[i] * np.abs(H[i, i])
                m = i

        # Record this value of m as prev_m for the next iteration.
        prev_m = m

        # Swap Rows & Columns (as appropriate)
        if m == n - 1:
            raise ValueError("ran out of row swaps")

        yy[m], yy[m + 1] = yy[m + 1], yy[m]

        H[[m, m + 1], :] = H[[m + 1, m], :]
        B[:, [m, m + 1]] = B[:, [m + 1, m]]

        # Remove corner on H diagonal
        # Variables named to be in keeping with the Ferguson, Bailey and Arno paper.
        # Note that the definitions fo these constants are modified to be correct *after* the row swap
        # (they are defined in terms of the pre-swapped matrix in the paper)
        if m < n - 2:
            delta = np.sqrt(
                H[m, m] * np.conj(H[m, m]) + H[m, m + 1] * np.conj(H[m, m + 1])
            )
            beta = H[m, m]
            lbd = H[m, m + 1]
            for i in range(m, n):
                # We update both H[i,m] and H[i,m+1], however each update requires the unupdated value of the other.
                # So we save the unmodified values before proceeding.
                t = H[i, m], H[i, m + 1]
                H[i, m] = t[0] * np.conj(beta) / delta + t[1] * np.conj(lbd) / delta
                H[i, m + 1] = -t[0] * lbd / delta + t[1] * beta / delta

        # Reduce H
        for i in range(m + 1, n):
            for j in reversed(range(min(i - 1, m + 1))):
                tf = AlgNearest(H[i, j] / H[j, j], D)
                yy[j] = yy[j] + tf * yy[i]

                for k in range(j):
                    H[i, k] = H[i, k] - tf * H[j, k]

                for k in range(n):
                    B[k, j] = B[k, j] + tf * B[k, i]

        # #NORM BOUND
        # M = 0
        # for j in range(n-1):
        # if np.abs(1/H[j,j]) > M:
        # M = np.abs(1/H[j,j])

        # Find the smallest magnitude yy value (which is our smallest linear combination value) and record it in the execution history.
        pos = np.argmin(np.abs(yy))
        miny = np.abs(yy[pos])

        candidateRelation = B[:, pos]

        # Check to see if the smallest linear combination is below our threshold for “0” (after normalisation of the B column vector)
        if miny / lin.norm(candidateRelation) < threshold:
            return candidateRelation

        # Check for diagonal element equal to “Zero”.  (I think this should be caught by the if block immediately above)
        if np.abs(H[n - 2, n - 2]) < threshold:
            print(
                "[WARNING]Found relation due to H[n-1,n-1]=0, which somehow got missed by the miny check."
            )
            return candidateRelation

        # H[i,i] = 0 should only be possible for i = n-1. However, *PERHAPS* it can happen for i ≠ n-1 due to unforseen numeric circumstances.
        for i in range(n - 2):
            if np.abs(H[i, i]) == 0.0:
                raise ValueError("Diagonal element of H is 0.")

    return


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
