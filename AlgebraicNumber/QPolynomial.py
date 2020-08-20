from fractions import Fraction
from math import gcd, factorial

import numpy.polynomial.polynomial as P
import numpy as np
import scipy.linalg as lin

from AlgebraicNumber.utils import nCr


class QPolynomial(object):
    """

    Examples:
      >>> p = QPolynomial()
      >>> p.printCoeff()
      '[]'
      >>> p = QPolynomial([1])
      >>> p.printCoeff()
      '[1]'
      >>> p = QPolynomial([-1,1])
      >>> p.printCoeff()
      '[-1,1]'
      >>> p = QPolynomial([-2,0,1])
      >>> p.printCoeff()
      '[-2,0,1]'
      >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
      >>> p.printCoeff()
      '[-2,0,2/3,0,1]'

    """

    __slots__ = ["__coeff", "F"]

    def __init__(self, coeff: list = [], field=Fraction):
        self.F = field
        self.__coeff = []
        self.__simplify(coeff)

    def __simplify(self, coeff):
        n = len(coeff)
        ns = None
        for i in reversed(range(n)):
            if coeff[i] != 0 and ns is None:
                ns = i + 1
                self.__coeff = [self.F(coeff[i])]
            elif not ns is None:
                self.__coeff = [self.F(coeff[i])] + self.__coeff

    def copy(self) -> "QPolynomial":
        p = self.__coeff.copy()
        res = QPolynomial(coeff=p, field=self.F)
        return res

    def __len__(self) -> int:
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> len(p)
          5

        """
        return len(self.__coeff)

    def __getitem__(self, i) -> "F":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p[2]
          Fraction(2, 3)
          >>> p[2:].printCoeff()
          '[2/3,0,1]'

        """
        n = len(self)

        if isinstance(i, int):
            return self.__coeff[i]
        elif isinstance(i, slice):
            i = list(range(*i.indices(n)))
        elif isinstance(i, (tuple, list)):
            pass

        n2 = len(i)

        res = []

        for r1, r2 in enumerate(i):
            res.append(self.__coeff[r2])

        return QPolynomial(coeff=res, field=self.F)

    def getCoefficients(self, conv=None) -> "conv":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.getCoefficients()
          [Fraction(-2, 1), Fraction(0, 1), Fraction(2, 3), Fraction(0, 1), Fraction(1, 1)]
          >>> p.getCoefficients(conv=float)
          [-2.0, 0.0, 0.66666..., 0.0, 1.0]

        """
        if conv is None:
            conv = self.F
        return [conv(x) for x in self.__coeff]

    def deg(self) -> int:
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.deg()
          4

        """
        return len(self.__coeff) - 1

    def __repr__(self):
        sp = ""
        sc = ""

        if self.deg() < 0:
            return "\n0"

        for p, c in enumerate(self.getCoefficients()):
            if p == 0 and c < 0:
                x = "-" + str(-c) + " "
                sc += x
                sp += " " * len(x)
            elif p == 0 and c == 0:
                pass
            elif p == 0 and c == 1:
                if len(sc) == 0:
                    x = "1 "
                else:
                    x = " "
                sc += x
                sp += " " * len(x)
            elif p == 0 and c > 0:
                x = str(c) + " "
                sc += x
                sp += " " * len(x)
            elif p == 1 and c < 0:
                if len(sc) == 0:
                    x = "-" + str(-c) + ".X "
                else:
                    x = "- " + str(-c) + ".X "
                sc += x
                sp += " " * len(x)
            elif p == 1 and c == 0:
                pass
            elif p == 1 and c == 1:
                if len(sc) == 0:
                    x = "X "
                else:
                    x = "+ X "
                sc += x
                sp += " " * len(x)
            elif p == 1 and c > 0:
                if len(sc) == 0:
                    x = str(c) + ".X "
                else:
                    x = "+ " + str(c) + ".X "
                sc += x
                sp += " " * len(x)
            elif p > 1 and c < 0:
                y = str(p)
                if len(sc) == 0:
                    x = "-" + str(-c) + ".X"
                else:
                    x = "- " + str(-c) + ".X"
                sc += x
                sp += " " * len(x)
                sc += " " * (len(y) + 1)
                sp += y + " "
            elif p > 1 and c == 0:
                pass
            elif p > 1 and c == 1:
                y = str(p)
                if len(sc) == 0:
                    x = "X"
                else:
                    x = "+ X"
                sc += x
                sp += " " * len(x)
                sc += " " * (len(y) + 1)
                sp += y + " "
            elif p > 1 and c > 0:
                y = str(p)
                if len(sc) == 0:
                    x = str(c) + ".X"
                else:
                    x = "+ " + str(c) + ".X"
                sc += x
                sp += " " * len(x)
                sc += " " * (len(y) + 1)
                sp += y + " "

        return sp + "\n" + sc

    def printCoeff(self) -> str:
        res = [str(x) for x in self.getCoefficients()]
        return "[" + ",".join(res) + "]"

    def __call__(self, x) -> float:
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p(0)
          -2.0...

        """
        return P.polyval(x, self.getCoefficients(conv=float))

    def __neg__(self) -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> (-p).printCoeff()
          '[2,0,-2/3,0,-1]'

        """
        res = []
        n = len(self)

        for i in range(n):
            r = -self[i]
            res.append(r)

        return QPolynomial(res, field=self.F)

    def __add__(self, b: "QPolynomial") -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p+q).printCoeff()
          '[-4,2/5,5/3,0,1]'

        """
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)

        res = []
        n = len(self)
        p = len(b)

        for i in range(max(n, p)):
            if i < n and i < p:
                r = self[i] + b[i]
            elif i < n and i >= p:
                r = self[i]
            elif i >= n and i < p:
                r = b[i]
            else:
                r = self.F(0)
            res.append(r)

        return QPolynomial(res, field=self.F)

    def __sub__(self, b: "QPolynomial") -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p-q).printCoeff()
          '[0,-2/5,-1/3,0,1]'
          >>> (p-p).printCoeff()
          '[]'

        """
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)

        mb = -b
        return self + mb

    def __truediv__(self, b: "QPolynomial") -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> (p/2).printCoeff()
          '[-1,0,1/3,0,1/2]'

        """
        res = []
        n = len(self)

        for i in range(n):
            r = self[i] / self.F(b)
            res.append(r)

        return QPolynomial(res, field=self.F)

    def __mul__(self, b: "QPolynomial") -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p*q).printCoeff()
          '[4,-4/5,-10/3,4/15,-4/3,2/5,1]'
          >>> (p*3).printCoeff()
          '[-6,0,2,0,3]'

        """
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)

        res = []
        n = len(self)
        p = len(b)

        for i in range(n + p + 1):
            r = self.F(0)
            for k in range(i + 1):
                j = i - k
                if k < n and j < p and j >= 0:
                    r += self[k] * b[j]
            res.append(r)

        return QPolynomial(res, field=self.F)

    def termWiseMul(self, b: "QPolynomial") -> "QPolynomial":
        """Hadamard product of the polynomials

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> p.termWiseMul(q).printCoeff()
          '[4,0,2/3]'

        """
        da = self.deg()
        db = b.deg()
        d = min(da, db)

        c = [self[i] * b[i] for i in range(d + 1)]

        return QPolynomial(coeff=c, field=self.F)

    def termWiseDiv(self, b: "QPolynomial") -> "QPolynomial":
        """Hadamard product of the polynomials

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> p.termWiseDiv(q).printCoeff()
          '[1,0,2/3]'

        """
        da = self.deg()
        db = b.deg()
        d = min(da, db)

        c = [self[i] / b[i] for i in range(d + 1)]

        return QPolynomial(coeff=c, field=self.F)

    def isNull(self) -> bool:
        """Checks if a polynomial is null

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.isNull()
          False
          >>> (p-p).isNull()
          True
          >>> p = QPolynomial()
          >>> p.isNull()
          True

        """
        return len(self) == 0

    def __eq__(self, b: "QPolynomial") -> bool:
        p = self - b
        return p.isNull()

    def __neq__(self, b: "QPolynomial") -> bool:
        return not self.__eq__(b)

    def integrate(self) -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.integrate().printCoeff()
          '[0,-2,0,2/9,0,1/5]'

        """
        res = [self.F(0)]
        n = len(self)

        for i in range(n):
            r = self[i] / self.F(i + 1)
            res.append(r)

        return QPolynomial(res, field=self.F)

    def derive(self) -> "QPolynomial":
        """

        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.derive().printCoeff()
          '[0,4/3,0,4]'

        """
        res = []
        n = len(self)

        for i in range(1, n):
            r = self[i] * self.F(i)
            res.append(r)

        return QPolynomial(res, field=self.F)

    def roots(self) -> np.array:
        """

        Examples:
          >>> p = QPolynomial([-2,0,1])
          >>> p.roots()
          array([-1.4142135...,  1.4142135...]...)

        """
        return P.polyroots(self.getCoefficients(conv=float))

    def squareFreeFact(self):
        r"""

        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
          >>> R.squareFreeFact().printCoeff()
          '[-1,0,0,0,1]'

        """
        n = self.deg()
        p = self.copy()

        while True:
            dp = p.derive()
            g = Qpolygcd(p, dp)
            if g == QPolynomial(coeff=[1], field=self.F):
                return p

            p, r = Qpolydiv(p, g)
            if not r.isNull():
                raise AssertionError(r)

    def compose(self, S: "QPolynomial") -> "QPolynomial":
        r"""Computes self(S(X))

        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 1])
          >>> S = QPolynomial(coeff=[-4, 5])
          >>> R.compose(S).printCoeff()
          '[21,-45,25]'

        """
        res = QPolynomial(field=self.F)
        sp = QPolynomial(coeff=[self.F(1)], field=self.F)
        for rk in self:
            if rk != 0:
                res = res + sp * rk
            sp = sp * S

        return res

    def translate(self, a: "QPolynomial") -> "QPolynomial":
        r"""

        Examples:
          >>> R = QPolynomial(coeff=[0, 0, 1])
          >>> R.translate(2).printCoeff()
          '[4,4,1]'

        """
        n = self.deg()

        q = (n + 1) * [None]
        for k in range(n + 1):
            q[k] = self.F(0)
            for p in range(k, n + 1):
                q[k] += nCr(p, k) * self[p] * a ** (p - k)

        return QPolynomial(coeff=q, field=self.F)

    def truncate(self, deg: int) -> "QPolynomial":
        """

        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
          >>> R.truncate(2).printCoeff()
          '[1,-1]'

        """
        p = self.getCoefficients()[: deg + 1]

        return QPolynomial(coeff=p, field=self.F)

    def reverse(self, deg: int = None) -> "QPolynomial":
        """This function returns the reverse polynomial associated with h, denoted rev(h)

        rev(h) is defined as follows:

        :math:`rev(h) = X^{deg(h)}.h(1/X)`

        If D is > deg(h), the reversed h will be multiplied by :math:`X^k` so that the final degree is D

        Returns:
          The reverse

        Examples:
          >>> h = QPolynomial(coeff=[1, 2, 3])
          >>> h.reverse().printCoeff()
          '[3,2,1]'
          >>> h = QPolynomial(coeff=[0, 1, 2, 3])
          >>> h.reverse().printCoeff()
          '[3,2,1]'

        """
        if deg is None:
            deg = self.deg()

        coeff = self.getCoefficients()[-1::-1]
        n = deg + 1 - len(coeff)
        assert n >= 0
        return QPolynomial(
            [self.F(0)] * n + self.getCoefficients()[-1::-1], field=self.F
        )

    def LogRev(self, D: int = None) -> "QPolynomial":
        """This function returns the logarithmic reverse rational power series associated with h, denoted LogRev(h)
        The result of this function is a truncature of degree D

        LogRev(h) is defined as follows :

        :math:`LogRev(h) = rev(h')/rev(h)`

        Args:
          D
            Degree of the resulting polynomial

        Returns:
          The logarithmic reverse

        Examples:
          >>> h = QPolynomial(coeff=[1, 2, 3])
          >>> h.LogRev().printCoeff()
          '[2,-2/3,-2/9]'
          >>> h.LogRev(D=5).printCoeff()
          '[2,-2/3,-2/9,10/27,-14/81,-2/243]'
          >>> h = QPolynomial(coeff=[0, -3])
          >>> h.LogRev().printCoeff()
          '[1]'

        """
        if D is None:
            D = self.deg()

        res = [self.F(0)] * (D + 1)

        for d in range(D + 1):
            s = self.newton_sum(d)
            res[d] = s

        return QPolynomial(coeff=res, field=self.F)

    def InvertLogRev(self) -> "QPolynomial":
        """Given a polynomial lr which is LogRev of h, finds back h

        Returns:
          The original polynomial

        Examples:
          >>> h = QPolynomial(coeff=[1, 2, 3])
          >>> l = h.LogRev()
          >>> l.InvertLogRev().printCoeff()
          '[1/3,2/3,1]'
          >>> h = QPolynomial(coeff=[-1, 0, 1])
          >>> l = h.LogRev(D=4)
          >>> l.printCoeff()
          '[2,0,2,0,2]'
          >>> l.InvertLogRev().printCoeff()
          '[-1,0,1]'
          >>> h = QPolynomial(coeff=[0, -3])
          >>> l = h.LogRev()
          >>> l.InvertLogRev().printCoeff()
          '[0,1]'

        """
        # D is the degree of the resulting h
        Df = float(self[0])
        Di = int(np.round(Df, 0))
        if abs(Di - Df) > 1e-6:
            raise AssertionError(self[0])

        # Computation of (D - LogRev(h))/X as the fraction n1, d1
        D = QPolynomial(coeff=[Di], field=self.F)
        p2 = (D - self)[1:]

        # Integrate p2
        p3 = p2.integrate()

        # Exponentiate p3
        p4 = QPolynomial(coeff=[1], field=self.F)
        for i in range(1, Di + 1):
            # dl : DL de exp(p3[i].x^i) a l'ordre D
            s = slice(0, Di + 1, i)
            s = list(range(*s.indices(Di + 1)))
            nc = 1 + Di // i

            dl = [self.F(0)] * (Di + 1)

            if i <= p3.deg():
                for u in range(nc):
                    dl[s[u]] = p3[i] ** u / factorial(u)
            else:
                dl[0] = self.F(1)

            dl = QPolynomial(coeff=dl, field=self.F)
            p4 = p4 * dl

        return p4.truncate(deg=Di).reverse(deg=Di)

    def companion(self) -> "QMatrix":
        """Computes the companion matrix of the polynomial

        Examples:
          >>> p = QPolynomial(coeff=[Fraction(3,4),Fraction(-2,7),Fraction(1,2)])
          >>> p.companion()
          [[  0, -3/4]
           [1/2,  2/7]]

        """
        from AlgebraicNumber.QMatrix import QMatrix

        n = len(self) - 1
        res = QMatrix.zeros((n, n), field=self.F)
        for i in range(n):
            res[i, -1] = -self[i]
            if i <= n - 2:
                res[i + 1, i] = self[-1]

        return res

    def discriminant(self) -> "F":
        """Computes the discriminant of the polynomial.
        The result is in the field

        Examples:
          >>> R = QPolynomial(coeff=[1, -3, 1, -3])
          >>> d = R.discriminant()
          >>> print(d)
          -400

        """
        n = self.deg()

        if ((n * (n - 1)) // 2) % 2 == 0:
            s = 1
        else:
            s = -1

        dp = self.derive()
        res = Qresultant(self, dp)

        dis = s * res / self[-1]

        return dis

    def mahler_separation_bound(self) -> float:
        r"""The minimum root separation for a polynomial P is defined as:

        .. math::
            sep(P) = min(|r-s|, (r,s) \in roots(P), r \neq s)

        *In the case of a square-free polynomial R of degree d and with integer coefficients*,
        this function gives a lower bound (Mahler, 1964) :

        .. math::
            sep(P) > \sqrt{\frac{3.|D|}{d^{d+2}}}. || R ||_2^{1-d}

        where D is the discriminant of the polynomial

        Examples:
          >>> R = QPolynomial(coeff=[1, -3, 1, -3])
          >>> b = R.mahler_separation_bound()
          >>> b
          0.111...

        """
        n = self.deg()

        d = float(self.discriminant())
        sep = np.sqrt(3 * np.abs(d) / n ** (n + 2)) / lin.norm(
            self.getCoefficients(conv=float), ord=2
        ) ** (n - 1)

        return sep

    def newton_sum(self, d: int) -> "F":
        """Computes the d-th Newton's sum :math:`s_d` of the polynomial h

        Given the roots :math:`x_k` of h :

        :math:`h(x_k) = 0`

        .. math::
            s_d=\sum_{k=1}^{n} x_k^d

        Args:
          d
            Order of the Newton's sum

        Returns:
          The d-th Newton's sum of h's roots

        Examples:
          >>> h = QPolynomial(coeff=[1, 2, 3])
          >>> for d in range(5): print(h.newton_sum(d))
          2
          -2/3
          -2/9
          10/27
          -14/81

        """
        from AlgebraicNumber.QMatrix import QMatrix, rowConcat

        Dh = self.deg()

        if d == 0:
            return self.F(Dh)

        # X is the vector of the d-th Newton's sum s_d :
        # with x_k a root of self,
        # s_d = x_1^d + x_2^d + ... + x_n^d
        # X = [s_1, s_2, ..., s_D]
        A = QMatrix.zeros((Dh, Dh), field=self.F)
        B = QMatrix.zeros((Dh, 1), field=self.F)

        for r in range(Dh):
            for u in range(r + 1):
                A[r, u] = self[Dh - r + u]

            B[r, 0] = -(r + 1) * self[-r - 2]

        X = A.inv() @ B

        if d <= Dh:
            return X[d - 1, 0]

        a_lrs = QMatrix.zeros((1, Dh), field=self.F)
        for u in range(Dh):
            a_lrs[0, u] = -self[u] / self[-1]

        for d in range(Dh + 1, d + 1):
            sd = a_lrs @ X

            X = rowConcat(X[1:, :], sd)

        return sd[0, 0]


def Qsylvester(R, S):
    """

    Examples:
      >>> R = QPolynomial(coeff=[1, -3, 1, -3, 1])
      >>> S = QPolynomial(coeff=[-3, 2, -9, 4])
      >>> m = Qsylvester(R, S)

    """
    from AlgebraicNumber.QMatrix import QMatrix

    m = R.deg()
    n = S.deg()

    res = QMatrix.zeros((n + m, n + m), field=R.F)

    for k in range(n):
        for p in range(1 + m):
            r = R[m - p]
            res[k, k + p] = r
    for k in range(m):
        for p in range(1 + n):
            s = S[n - p]
            res[k + n, k + p] = s

    return res


def Qresultant(R, S):
    """

    Examples:
      >>> R = QPolynomial(coeff=[1, -3, 1, -3, 1])
      >>> S = QPolynomial(coeff=[-3, 2, -9, 4])
      >>> print(Qresultant(R, S))
      -4563

    """
    m = Qsylvester(R, S)
    return m.det()


def Qpolydiv(n, d):
    """

    Examples:
      >>> n = QPolynomial(coeff=[-4, 0, -2, 1])
      >>> d = QPolynomial(coeff=[-3, 1])
      >>> q,r = Qpolydiv(n, d)
      >>> q.printCoeff()
      '[3,1,1]'
      >>> r.printCoeff()
      '[5]'

    """
    F = n.F

    if d.isNull():
        raise ZeroDivisionError

    q = QPolynomial(field=F)
    r = n.copy()  # At each step n = d × q + r

    while not r.isNull() and r.deg() >= d.deg():
        c = r[-1] / d[-1]  # Divide the leading terms
        t = QPolynomial(coeff=(r.deg() - d.deg()) * [F(0)] + [c], field=F)
        q = q + t
        r = r - t * d

    return (q, r)


def Qpolygcd(a, b):
    r"""

    Examples:
      >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
      >>> S = QPolynomial(coeff=[-1, 0, 0, -4, 5])
      >>> g = Qpolygcd(S, R)
      >>> g.printCoeff()
      '[-1,1]'

    """
    m = a.deg()
    n = b.deg()

    if n > m:
        a, b = b, a
        n, m = m, n

    # Here, deg(a) >= deb(b)
    while True:
        q, r = Qpolydiv(a, b)

        a, b = b, r

        if b.isNull():
            return a / a[-1]


def Qnpolymul(*polynomials):
    lp = list(polynomials)

    if len(lp) == 0:
        return QPolynomial(coeff=[1])

    F = lp[0].F
    res = QPolynomial(coeff=[F(1)], field=F)
    for q in lp:
        res = res * q

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
