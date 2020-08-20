from fractions import Fraction
from math import factorial


class QMatrix(object):
    """

    Examples:
      >>> m = QMatrix(coeff=[[0,10,0],[0,0,0],[0,0,0]])
      >>> m
      [[0, 10, 0]
       [0,  0, 0]
       [0,  0, 0]]
      >>> print(m[0,1])
      10
      >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
      >>> p
      [[3, -2, 1]
       [4,  7, 2]]

    """

    @classmethod
    def zeros(cls, shp, field=Fraction):
        n, m = shp

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = field(0)

        return QMatrix(coeff=res, field=field)

    @classmethod
    def ones(cls, shp, field=Fraction):
        n, m = shp

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = field(1)

        return QMatrix(coeff=res, field=field)

    @classmethod
    def eye(cls, n, field=Fraction):
        res = []
        for i in range(n):
            res.append([field(0)] * n)

        for i in range(n):
            res[i][i] = field(1)

        return QMatrix(coeff=res, field=field)

    def __init__(self, coeff: list = [], field=Fraction):
        self.F = field
        self.__coeff = self.__simplify(coeff)

    def __simplify(self, coeff):
        n = len(coeff)
        if n > 0:
            m = len(coeff[0])
        else:
            m = 0

        res = []
        row = [None] * m
        for i in range(n):
            res.append(row.copy())

        for i in range(n):
            for j in range(m):
                res[i][j] = self.F(coeff[i][j])

        return res

    def __repr__(self):
        n, m = self.shape

        res = []
        for i in range(n):
            res.append([None] * m)

        for c in range(m):
            lc = 0
            for r in range(n):
                lc = max(lc, len(str(self[r, c])))
            for r in range(n):
                s = str(self[r, c])
                res[r][c] = (lc - len(s)) * " " + s

        txt = "[["
        for r in range(n - 1):
            txt += ", ".join(res[r])
            txt += "]\n ["
        txt += ", ".join(res[-1])
        txt += "]]"

        return txt

    @property
    def shape(self) -> tuple:
        n = len(self.__coeff)
        if n > 0:
            m = len(self.__coeff[0])
        else:
            m = 0

        return n, m

    def copy(self) -> "QMatrix":
        n, m = self.shape

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = self[i, j]

        return QMatrix(coeff=res, field=self.F)

    def __getitem__(self, key: tuple) -> "F":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> print(p[1,1])
          7
          >>> p[:,1]
          [[-2]
           [ 7]]
          >>> p[1,:]
          [[4, 7, 2]]
          >>> p[[1,0],:]
          [[4,  7, 2]
           [3, -2, 1]]

        """
        n, m = self.shape

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return self.__coeff[i][j]

        if isinstance(i, int):
            i = [i]
        elif isinstance(i, slice):
            i = list(range(*i.indices(n)))
        elif isinstance(i, (tuple, list)):
            pass

        if isinstance(j, int):
            j = [j]
        elif isinstance(j, slice):
            j = list(range(*j.indices(m)))
        elif isinstance(j, (tuple, list)):
            pass

        n2 = len(i)
        m2 = len(j)

        res = []
        for _ in range(n2):
            res.append([None] * m2)

        for r1, r2 in enumerate(i):
            for s1, s2 in enumerate(j):
                res[r1][s1] = self.__coeff[r2][s2]

        return QMatrix(coeff=res, field=self.F)

    def __setitem__(self, key: tuple, val: "F"):
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> q = QMatrix(coeff=[[0,0,1]])
          >>> p[1,-1] = 10
          >>> p
          [[3, -2,  1]
           [4,  7, 10]]
          >>> p[1,:] = q
          >>> p
          [[3, -2, 1]
           [0,  0, 1]]
          >>> p[[1,0],:] = p[[0,1],:]
          >>> p
          [[0,  0, 1]
           [3, -2, 1]]

        """
        n, m = self.shape

        i, j = key

        if isinstance(i, int) and isinstance(j, int):
            self.__coeff[i][j] = val
            return

        if isinstance(i, int):
            i = [i]
        elif isinstance(i, slice):
            i = list(range(*i.indices(n)))
        elif isinstance(i, (tuple, list)):
            pass

        if isinstance(j, int):
            j = [j]
        elif isinstance(j, slice):
            j = list(range(*j.indices(m)))
        elif isinstance(j, (tuple, list)):
            pass

        n2 = len(i)
        m2 = len(j)

        res = []
        for _ in range(n2):
            res.append([None] * m2)

        for r1, r2 in enumerate(i):
            for s1, s2 in enumerate(j):
                self.__coeff[r2][s2] = val[r1, s1]

    def __truediv__(self, b: "F") -> "QMatrix":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> p/3
          [[  1, -2/3, 1/3]
           [4/3,  7/3, 2/3]]

        """
        n, m = self.shape

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = self[i, j] / self.F(b)

        return QMatrix(coeff=res, field=self.F)

    def __add__(self, b: "QMatrix") -> "QMatrix":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> q = QMatrix(coeff=[[5,1,2],[3,3,4]])
          >>> p+q
          [[8, -1, 3]
           [7, 10, 6]]

        """
        n, m = self.shape
        if n != b.shape[0] or m != b.shape[1]:
            raise AssertionError(n, m, b.shape)

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = self[i, j] + b[i, j]

        return QMatrix(coeff=res, field=self.F)

    def __neg__(self) -> "QMatrix":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> -p
          [[-3,  2, -1]
           [-4, -7, -2]]

        """
        n, m = self.shape

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = -self[i, j]

        return QMatrix(coeff=res, field=self.F)

    def __mul__(self, b: "F") -> "QMatrix":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> p*3
          [[ 9, -6, 3]
           [12, 21, 6]]

        """
        n, m = self.shape

        res = []
        for i in range(n):
            res.append([None] * m)

        for i in range(n):
            for j in range(m):
                res[i][j] = self[i, j] * self.F(b)

        return QMatrix(coeff=res, field=self.F)

    def __sub__(self, b: "QMatrix") -> "QMatrix":
        """

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> q = QMatrix(coeff=[[5,1,2],[3,3,4]])
          >>> p-q
          [[-2, -3, -1]
           [ 1,  4, -2]]
          >>> p-p
          [[0, 0, 0]
           [0, 0, 0]]

        """
        mb = -b

        return self + mb

    def __matmul__(self, b):
        """

        Examples:
          >>> p = QMatrix(coeff=[[1,49,-7], [0,-69,103],[-4,-4,-6]])
          >>> q = QMatrix(coeff=[[1,10,1],[0,-17,131],[8,-5,-1]])
          >>> p@q
          [[-55, -788,  6427]
           [824,  658, -9142]
           [-52,   58,  -522]]

        """
        n, m = self.shape
        nb, p = b.shape

        if m != nb:
            raise AssertionError(n, m, nb, p)

        res = QMatrix.zeros((n, p), field=self.F)

        for i in range(n):
            for j in range(p):
                for k in range(m):
                    res[i, j] = res[i, j] + self[i, k] * b[k, j]

        return res

    def isNull(self) -> bool:
        """Checks if a matrix is null

        Examples:
          >>> p = QMatrix(coeff=[[3,-2,1],[4,7,2]])
          >>> p.isNull()
          False
          >>> (p-p).isNull()
          True
          >>> p = QMatrix()
          >>> p.isNull()
          True

        """
        n, m = self.shape

        res = True
        for i in range(n):
            for j in range(m):
                if self[i, j] != self.F(0):
                    res = False

        return res

    def __eq__(self, b: "QMatrix") -> bool:
        p = self - b
        return p.isNull()

    def __neq__(self, b: "QMatrix") -> bool:
        return not self.__eq__(b)

    def GaussJordan(self, debug=False):
        """

        Examples:
          >>> p = QMatrix(coeff=[[2,-1,0],[0,-1,2],[-1,2,-1]])
          >>> s,pivots,PM = p.GaussJordan()
          >>> s
          -1
          >>> pivots
          [Fraction(2, 1), Fraction(3, 2), Fraction(4, 3)]
          >>> PM
          [[1, 0, 0]
           [0, 1, 0]
           [0, 0, 1]]

        """
        n, m = self.shape
        PM = self.copy()

        pivots = []
        s = 1

        for r in range(n):  # r décrit tous les indices de colonnes
            # Rechercher max(|A[i,r]|, r+1 ≤ i ≤ n).
            # Noter k l'indice de ligne du maximum
            # => A[k,r] est le pivot
            if r == n - 1:
                k = r
            else:
                k = r
                km = abs(PM[r, r])
                for i in range(r, n):
                    if abs(PM[i, r]) > km:
                        km = abs(PM[i, r])
                        k = i

            piv = PM[k, r]

            if debug:
                print("piv", float(piv))

            pivots.append(piv)

            # Diviser la ligne k par A[k,r] de façon que le pivot prenne la valeur 1
            for u in range(m):
                PM[k, u] = PM[k, u] / piv

            if k != r:
                s *= -1
                # Échanger les lignes k et r  (On place la ligne du pivot en position r)
                PM[[k, r], :] = PM[[r, k], :]

            for i in range(n):  # On simplifie les autres lignes
                if i != r:
                    # Soustraire à la ligne i la ligne r multipliée par A[i,r]
                    # (de façon à annuler A[i,r])
                    # AM[i,u] = AM[i,u] - AM[r,u]*AM[i,r]

                    if debug:
                        print("i,dL", i, PM[i, :] - PM[r, :] * PM[i, r])
                    PM[i, :] = PM[i, :] - PM[r, :] * PM[i, r]
                    if debug:
                        print()

                        print("P", PM)
                        print()

            if debug:
                print("pivots", [float(p) for p in pivots])
                print(72 * "=")

        return s, pivots, PM

    def inv(self, debug=False):
        """

        Examples:
          >>> p = QMatrix(coeff=[[2,-5,-5], [7,-1,0],[-8,3,2]])
          >>> p.inv()
          [[ -2,  -5,  -5]
           [-14, -36, -35]
           [ 13,  34,  33]]

        """
        n, m = self.shape

        if n != m:
            raise AssertionError(n, m)

        Mi = colConcat(self, QMatrix.eye(n, self.F))
        s, pivots, R = Mi.GaussJordan(debug)

        return R[:, n:]

    def det(self, debug=False):
        """

        Examples:
          >>> p = QMatrix(coeff=[[2,-5,-5], [7,-1,0],[-8,3,2]])
          >>> print(p.det())
          1

        """
        s, pivots, R = self.GaussJordan(debug)
        res = s
        for a in pivots:
            res = a * res
        return res


def colConcat(A: QMatrix, B: QMatrix) -> QMatrix:
    n, m = A.shape
    nb, p = B.shape

    if n == 0 and m == 0:
        return B.copy()

    if nb == 0 and p == 0:
        return A.copy()

    if n != nb:
        raise AssertionError(n, m, nb, p)

    res = QMatrix.zeros((n, m + p), field=A.F)

    for i in range(n):
        for j in range(m):
            res[i, j] = A[i, j]

    for i in range(nb):
        for j in range(p):
            res[i, n + j] = B[i, j]

    return res


def rowConcat(A: QMatrix, B: QMatrix) -> QMatrix:
    n, m = A.shape
    nb, p = B.shape

    if n == 0 and m == 0:
        return B.copy()

    if nb == 0 and p == 0:
        return A.copy()

    if m != p:
        raise AssertionError(n, m, nb, p)

    res = QMatrix.zeros((n + nb, m), field=A.F)

    for i in range(n):
        for j in range(m):
            res[i, j] = A[i, j]

    for i in range(nb):
        for j in range(p):
            res[m + i, j] = B[i, j]

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
