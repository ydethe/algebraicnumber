from fractions import Fraction
from math import gcd

from numpy.polynomial import polynomial as P
import numpy as np
from numpy.ma import masked_array
import scipy.linalg as lin

from AlgebraicNumber.utils import prod


class QMatrix(object):
    @classmethod
    def eye(cls, n):
        return QMatrix(P=np.eye(n, dtype=np.int64), Q=np.ones((n, n), dtype=np.int64))

    @classmethod
    def zeros(cls, n, m):
        return QMatrix(
            P=np.zeros((n, m), dtype=np.int64), Q=np.ones((n, m), dtype=np.int64)
        )

    @classmethod
    def ones(cls, n, m):
        return QMatrix(
            P=np.ones((n, m), dtype=np.int64), Q=np.ones((n, m), dtype=np.int64)
        )

    def __init__(self, P=None, Q=None):
        if not P is None and not Q is None:
            self.__p_coeff = np.int64(np.round(P, 0))
            self.__q_coeff = np.int64(np.round(Q, 0))
        elif not P is None and Q is None:
            n, m = P.shape
            self.__p_coeff = np.int64(np.round(P, 0))
            self.__q_coeff = np.ones((n, m), dtype=np.int64)
        elif P is None and Q is None:
            raise AssertionError("P et Q are None")

        self.__simplify()

    def __simplify(self):
        # Reducing the coefficients
        n, m = self.__p_coeff.shape
        for i in range(n):
            for j in range(m):
                p = self.__p_coeff[i, j]
                q = self.__q_coeff[i, j]
                d = gcd(p, q)
                if q < 0:
                    d *= -1
                self.__p_coeff[i, j] = p // d
                self.__q_coeff[i, j] = q // d

    def asFloat(self):
        return self.__p_coeff / self.__q_coeff

    def getCoefficientsAsNumDen(self):
        """Returns the degree of the polynomial
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> a,b = p.getCoefficientsAsNumDen()
          >>> a
          array([[  1,  49,  -7],
                 [  0, -69, 103],
                 [ -2,  -4,  -6]]...
          >>> b
          array([[  1,  16,  12],
                 [  1,  40, 120],
                 [  3,   9,   5]]...
                  
        """
        return self.__p_coeff.copy(), self.__q_coeff.copy()

    def __getitem__(self, key):
        p = self.__p_coeff[key]
        q = self.__q_coeff[key]

        ki, kj = key
        if isinstance(ki, int) and isinstance(kj, int):
            res = Fraction(p, q)
        elif not isinstance(ki, int) and isinstance(
            kj, int
        ):  # Selection of the column kj
            n = len(p)
            res = QMatrix(p.reshape((n, 1)), q.reshape((n, 1)))
        elif isinstance(ki, int) and not isinstance(kj, int):  # Selection of the row ki
            n = len(p)
            res = QMatrix(p.reshape((1, n)), q.reshape((1, n)))
        else:
            res = QMatrix(p, q)

        return res

    def __setitem__(self, i, val):
        self.__p_coeff[i] = val.numerator
        self.__q_coeff[i] = val.denominator

    def __repr__(self):
        s = str(self.__p_coeff)
        s += "\n"
        s += str(self.__q_coeff)
        return s

    @property
    def shape(self):
        return self.__p_coeff.shape

    def __add__(self, b):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> q = QMatrix(P=[[1,10,1],[0,-17,131],[8,-5,-1]], Q=[[1,81,24],[1,216,192],[6,9,2]])
          >>> p+q
          [[   2 4129  -13]
           [   0 -487  493]
           [   2   -1  -17]]
          [[   1 1296   24]
           [   1  270  320]
           [   3    1   10]]
          
        """
        if not isinstance(b, QMatrix) and not hasattr(b, "__iter__"):
            p0 = b.numerator
            q0 = b.denominator
            b = QMatrix(
                P=p0 * np.eye(n, dtype=np.int64), Q=q0 * np.ones((n, n), dtype=np.int64)
            )

        n, m = self.shape
        nb, mb = b.shape

        if n != nb or m != mb:
            raise AssertionError(n, m, nb, mb)

        P = np.empty((n, m), dtype=np.int64)
        Q = np.empty((n, m), dtype=np.int64)

        for i in range(n):
            for j in range(m):
                pc = self[i, j] + b[i, j]

                P[i, j] = pc.numerator
                Q[i, j] = pc.denominator

        res = QMatrix(P, Q)

        return res

    def __mul__(self, b):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> q = QMatrix(P=[[1,10,1],[0,-17,131],[8,-5,-1]], Q=[[1,81,24],[1,216,192],[6,9,2]])
          >>> p*q
          [[    1   245    -7]
           [    0   391 13493]
           [   -8    20     3]]
          [[    1   648   288]
           [    1  2880 23040]
           [    9    81     5]]
          
        """
        if not isinstance(b, QMatrix) and not hasattr(b, "__iter__"):
            p0 = b.numerator
            q0 = b.denominator
            b = QMatrix(
                P=p0 * np.eye(n, dtype=np.int64), Q=q0 * np.ones((n, n), dtype=np.int64)
            )

        n, m = self.shape
        nb, mb = b.shape

        if n != nb or m != mb:
            raise AssertionError(n, m, nb, mb)

        P = np.empty((n, m), dtype=np.int64)
        Q = np.empty((n, m), dtype=np.int64)

        for i in range(n):
            for j in range(m):
                pc = self[i, j] * b[i, j]

                P[i, j] = pc.numerator
                Q[i, j] = pc.denominator

        res = QMatrix(P, Q)

        return res

    def __neg__(self):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> -p
          [[  -1  -49    7]
           [   0   69 -103]
           [   2    4    6]]
          [[  1  16  12]
           [  1  40 120]
           [  3   9   5]]
          
        """
        n, m = self.shape

        P = np.empty((n, m), dtype=np.int64)
        Q = np.empty((n, m), dtype=np.int64)

        for i in range(n):
            for j in range(m):
                pa = self[i, j].numerator
                qa = self[i, j].denominator

                P[i, j] = -pa
                Q[i, j] = qa

        res = QMatrix(P, Q)

        return res

    def __sub__(self, b):
        nb = -b
        return self + nb

    def __matmul__(self, b):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> q = QMatrix(P=[[1,10,1],[0,-17,131],[8,-5,-1]], Q=[[1,81,24],[1,216,192],[6,9,2]])
          >>> p@q
          [[    2  2141  2481]
           [  103 -2947 -2467]
           [  -34   301   581]]
          [[    9 10368  1024]
           [   90  8640  1536]
           [   15   486  2160]]
          
        """
        if not isinstance(b, QMatrix) and not hasattr(b, "__iter__"):
            p0 = b.numerator
            q0 = b.denominator
            P, Q = self.getCoefficientsAsNumDen()
            res = QMatrix(P=p0 * P, Q=q0 * Q)
            return res

        n, m = self.shape
        nb, p = b.shape

        if m != nb:
            raise AssertionError(n, m, nb, p)

        P = np.empty((n, p), dtype=np.int64)
        Q = np.empty((n, p), dtype=np.int64)

        for i in range(n):
            for j in range(p):
                c = Fraction(0, 1)
                for k in range(m):
                    c = c + self[i, k] * b[k, j]

                pc = c.numerator
                qc = c.denominator

                P[i, j] = pc
                Q[i, j] = qc

        res = QMatrix(P, Q)

        return res

    def isNull(self) -> bool:
        """Checks if a matrix is null
        
        """
        return np.all(self.__p_coeff == 0)

    def __eq__(self, b):
        p = self - b
        return p.isNull()

    def __neq__(self, b):
        return not self.__eq__(b)

    def GaussJordan(self, debug=False):
        n, m = self.shape
        PM, QM = self.getCoefficientsAsNumDen()

        pivots = []
        s = 1

        for r in range(n):  # r décrit tous les indices de colonnes
            # Rechercher max(|A[i,r]|, r+1 ≤ i ≤ n).
            # Noter k l'indice de ligne du maximum
            # => A[k,r] est le pivot
            if r == n - 1:
                k = r
            else:
                v = np.ones(n, dtype=bool)
                for i in range(r, n):
                    if QM[i, r] != 0:
                        v[i] = False
                Pp = masked_array(PM[:, r], v)
                Qp = masked_array(QM[:, r], v)
                k = np.argmax(np.abs(Pp / Qp))

            if PM[k, r] == 0 or QM[k, r] == 0:
                return Fraction(0, 1)

            piv = Fraction(PM[k, r], QM[k, r])

            if debug:
                print("piv", float(piv))

            pivots.append(piv)

            # Diviser la ligne k par A[k,r] de façon que le pivot prenne la valeur 1
            QM[k, :] *= piv.numerator
            PM[k, :] *= piv.denominator
            for u in range(m):
                d = gcd(PM[k, u], QM[k, u])
                if QM[k, u] * d < 0:
                    d *= -1
                PM[k, u] = PM[k, u] // d
                QM[k, u] = QM[k, u] // d

            if k != r:
                s *= -1
                # Échanger les lignes k et r  (On place la ligne du pivot en position r)
                PM[[k, r], :] = PM[[r, k], :]
                QM[[k, r], :] = QM[[r, k], :]

            for i in range(n):  # On simplifie les autres lignes
                if i != r:
                    # Soustraire à la ligne i la ligne r multipliée par A[i,r]
                    # (de façon à annuler A[i,r])
                    # AM[i,u] = AM[i,u] - AM[r,u]*AM[i,r]

                    # PM[i,u]   PM[i,u]   PM[r,u]*PM[i,r]
                    # ------- = ------- - ---------------
                    # QM[i,u]   QM[i,u]   QM[r,u]*QM[i,r]
                    if debug:
                        print(
                            "i,dL",
                            i,
                            PM[i, :] / QM[i, :]
                            - PM[r, :] / QM[r, :] * PM[i, r] / QM[i, r],
                        )
                    up = PM[i, :] * QM[r, :] * QM[i, r] - QM[i, :] * PM[r, :] * PM[i, r]
                    uq = QM[i, :] * QM[r, :] * QM[i, r]
                    for u in range(m):
                        d = gcd(up[u], uq[u])
                        if uq[u] * d < 0:
                            d *= -1
                        if debug:
                            print("%i/%i," % (up[u] // d, uq[u] // d), end="")
                        if d == 0:
                            if debug:
                                print("k,i,r", k, i, r)
                                print(PM[i, u], QM[i, u])
                                print(PM[r, u], QM[r, u])
                                print(PM[i, r], QM[i, r])
                                print(up[u], uq[u])
                            raise ZeroDivisionError
                        PM[i, u] = up[u] // d
                        QM[i, u] = uq[u] // d
                    if debug:
                        print()

                        print("P", PM)
                        print("Q", QM)
                        print()

            if debug:
                print("pivots", [float(p) for p in pivots])
                print(72 * "=")

        return s, pivots, QMatrix(PM, QM)

    def inv(self, debug=False):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> p.inv()
          [[105904 169960 140175]
           [-24720 -68640 -37080]
           [ -2160  -3000  -3240]]
          [[ 59179  59179 118358]
           [ 59179  59179  59179]
           [  2573   2573   2573]]
          
        """
        n, m = self.shape

        if n != m:
            raise AssertionError(n, m)

        P, Q = self.getCoefficientsAsNumDen()

        Pi = np.hstack((P, np.eye(n, dtype=np.int64)))
        Qi = np.hstack((Q, np.ones((n, n), dtype=np.int64)))
        Mi = QMatrix(Pi, Qi)
        s, pivots, R = Mi.GaussJordan(debug)

        PM, QM = R.getCoefficientsAsNumDen()

        return QMatrix(PM[:, n:], QM[:, n:])

    def det(self, debug=False):
        """
        
        Examples:
          >>> p = QMatrix(P=[[1,49,-7], [0,-69,103],[-4,-4,-6]], Q=[[1,16,12],[1,40,120],[6,9,5]])
          >>> print(p.det())
          59179/43200
          
        """
        s, pivots, R = self.GaussJordan(debug)
        return s * prod(*pivots)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
