from fractions import Fraction
from math import gcd, factorial

from numpy.polynomial import polynomial as P
import numpy as np
from numpy.ma import masked_array
import scipy.linalg as lin

from AlgebraicNumber.utils import nCr


class QPolynomial(object):
    r"""Class the represents polynomial with rational coefficients
    
    Most implemented methods work with rational numbers, so that their result is exact.
    
    """

    def __init__(self, coeff=None, p_coeff=None, q_coeff=None, check=True):
        if not coeff is None:
            if check:
                for c in coeff:
                    if not isinstance(c, Fraction):
                        raise AssertionError("Not a Fraction", c)
            self.__p_coeff = np.int64([x.numerator for x in coeff])
            self.__q_coeff = np.int64([x.denominator for x in coeff])
        elif not p_coeff is None and not q_coeff is None:
            self.__p_coeff = np.int64(np.round(p_coeff, 0))
            self.__q_coeff = np.int64(np.round(q_coeff, 0))
        elif not p_coeff is None and q_coeff is None:
            n = len(p_coeff)
            self.__p_coeff = np.int64(np.round(p_coeff, 0))
            self.__q_coeff = np.ones(n, dtype=np.int64)
        elif p_coeff is None and not q_coeff is None:
            n = len(q_coeff)
            self.__p_coeff = np.ones(n, dtype=np.int64)
            self.__q_coeff = np.int64(np.round(q_coeff, 0))
        elif p_coeff is None and q_coeff is None:
            self.__p_coeff = np.array([], dtype=np.int64)
            self.__q_coeff = np.array([], dtype=np.int64)

        self.__simplify()

    def __simplify(self):
        # Trimming the null high order coefficients
        n = len(self.__p_coeff)
        for i in reversed(range(n)):
            if self.__p_coeff[i] != 0:
                self.__p_coeff = self.__p_coeff[: i + 1]
                self.__q_coeff = self.__q_coeff[: i + 1]
                break
            elif i == 0:
                self.__p_coeff = np.array([], dtype=np.int64)
                self.__q_coeff = np.array([], dtype=np.int64)

        # Reducing the coefficients
        n = len(self.__p_coeff)
        for i in range(n):
            p = self.__p_coeff[i]
            q = self.__q_coeff[i]
            d = gcd(p, q)
            if q < 0:
                d *= -1
            self.__p_coeff[i] = p // d
            self.__q_coeff[i] = q // d

        if len(self.__p_coeff) > 0 and self.__p_coeff[-1] == 0:
            raise AssertionError()

    def getCoefficientsAsFraction(self):
        """Returns the degree of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> print(p.getCoefficientsAsFraction())
          [Fraction(3, 4), Fraction(-2, 7), Fraction(1, 2)]
          
        """
        res = [Fraction(p, q) for (p, q) in zip(self.__p_coeff, self.__q_coeff)]
        return res

    def getCoefficientsAsNumDen(self):
        """Returns the degree of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> print(p.getCoefficientsAsNumDen())
          (array([ 3, -2,  1]..., array([4, 7, 2]...)
          
        """
        return self.__p_coeff.copy(), self.__q_coeff.copy()

    def __repr__(self):
        s = str(self.__p_coeff)
        s += "\n"
        s += str(self.__q_coeff)
        return s

    def deg(self):
        """Returns the degree of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> p.deg()
          2
          
        """
        return len(self) - 1

    def __len__(self):
        return len(self.__p_coeff)

    def __getitem__(self, n):
        p = self.__p_coeff[n]
        q = self.__q_coeff[n]

        res = Fraction(p, q)

        return res

    def __truediv__(self, b):
        if b == 0:
            raise ZeroDivisionError

        if isinstance(b, Fraction):
            p = b.numerator
            q = b.denominator
        else:
            p = np.int64(b)
            q = 1

        res = QPolynomial(p_coeff=self.__p_coeff * q, q_coeff=self.__q_coeff * p)

        return res

    def __add__(self, b):
        """
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> q = QPolynomial(p_coeff=[1,-2], q_coeff=[4,2])
          >>> p+q
          [ 1 -9  1]
          [1 7 2]
          >>> p = QPolynomial(p_coeff=[], q_coeff=[])
          >>> p+q
          [ 1 -1]
          [4 1]
          
        """
        if not isinstance(b, QPolynomial) and not hasattr(b, "__iter__"):
            b = QPolynomial(coeff=[b])

        na = len(self)
        nb = len(b)

        n = max(na, nb)
        p_res = np.empty(n)
        q_res = np.empty(n)

        for i in range(n):
            if i < na and i < nb:
                r = self[i] + b[i]
            elif i < na and i >= nb:
                r = self[i]
            elif i >= na and i < nb:
                r = b[i]
            else:
                r = Fraction(0, 1)

            p_res[i] = r.numerator
            q_res[i] = r.denominator

        res = QPolynomial(p_coeff=p_res, q_coeff=q_res)

        return res

    def __neg__(self):
        """
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> -p
          [-3  2 -1]
          [4 7 2]
          
        """
        res = QPolynomial(p_coeff=-self.__p_coeff, q_coeff=self.__q_coeff)

        return res

    def __sub__(self, b):
        """
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> q = QPolynomial(p_coeff=[1,-2], q_coeff=[4,2])
          >>> p-q
          [1 5 1]
          [2 7 2]
          >>> p-p
          []
          []
          
        """
        if not isinstance(b, QPolynomial) and not hasattr(b, "__iter__"):
            b = QPolynomial(coeff=[b])

        mb = -b

        return self + mb

    def __mul__(self, b):
        """
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> q = QPolynomial(p_coeff=[1,-2], q_coeff=[4,2])
          >>> p*q
          [  3 -23  23  -1]
          [16 28 56  2]
          >>> p = QPolynomial(p_coeff=[], q_coeff=[])
          >>> p*q
          []
          []
          
        """
        if not isinstance(b, QPolynomial) and not hasattr(b, "__iter__"):
            b = QPolynomial(coeff=[b])

        na = len(self)
        nb = len(b)

        n = na + nb
        p_res = np.empty(n)
        q_res = np.empty(n)

        for i in range(n):
            r = Fraction(0, 1)
            for k in range(i + 1):
                if k < na and i - k < nb and i - k >= 0:
                    r += self[k] * b[i - k]

            p_res[i] = r.numerator
            q_res[i] = r.denominator

        res = QPolynomial(p_coeff=p_res, q_coeff=q_res)

        return res

    def termWiseMul(self, b):
        """Hadamard product of the polynomials
        
        """
        da = self.deg()
        db = b.deg()
        d = min(da, db)

        c = [self[i] * b[i] for i in range(d + 1)]

        return QPolynomial(coeff=c)

    def termWiseDiv(self, b):
        """Hadamard product of the polynomials
        
        """
        da = self.deg()
        db = b.deg()
        d = min(da, db)

        c = [self[i] / b[i] for i in range(d + 1)]

        return QPolynomial(coeff=c)

    def __call__(self, x):
        """
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> p(3)
          4.392857142...
          
        """
        a = P.polyval(x, self.__p_coeff / self.__q_coeff)
        return a

    def isNull(self) -> bool:
        """Checks if a polynomial is null
        
        """
        return len(self) == 0

    def __eq__(self, b):
        p = self - b
        return p.isNull()

    def __neq__(self, b):
        return not self.__eq__(b)

    def integrate(self) -> "QPolynomial":
        """Computes the integral of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> p.integrate()
          [ 0  3 -1  1]
          [1 4 7 6]
          >>> p = QPolynomial(p_coeff=[], q_coeff=[])
          >>> p.integrate()
          []
          []
          >>> p = QPolynomial(p_coeff=[1], q_coeff=[1])
          >>> p.integrate()
          [0 1]
          [1 1]
          
        """
        p = np.hstack(([0], self.__p_coeff))
        q = np.hstack(([1], self.__q_coeff * np.arange(1, len(self) + 1)))

        res = QPolynomial(p_coeff=p, q_coeff=q)

        return res

    def derive(self) -> "QPolynomial":
        """Computes the derivative of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> p.derive()
          [-2  1]
          [7 1]
          >>> p = QPolynomial(p_coeff=[], q_coeff=[])
          >>> p.derive()
          []
          []
          >>> p = QPolynomial(p_coeff=[1], q_coeff=[1])
          >>> p.derive()
          []
          []
          
        """
        p = self.__p_coeff[1:] * np.arange(1, len(self))
        q = self.__q_coeff[1:]

        res = QPolynomial(p_coeff=p, q_coeff=q)

        return res

    def companion(self) -> np.array:
        """Computes the companion matrix of the polynomial
        
        Examples:
          >>> p = QPolynomial(p_coeff=[3,-2,1], q_coeff=[4,7,2])
          >>> p.companion()
          [[ 0 -3]
           [ 1  2]]
          [[1 4]
           [2 7]]
          
        """
        from AlgebraicNumber.QMatrix import QMatrix

        n = len(self) - 1
        res = QMatrix.zeros(n, n)
        for i in range(n):
            res[i, -1] = -self[i]
            if i <= n - 2:
                res[i + 1, i] = self[-1]

        return res

    def discriminant(self) -> Fraction:
        """Computes the discriminant of the polynomial.
        The result is a fraction
        
        Examples:
          >>> R = QPolynomial(p_coeff=[1, -3, 1, -3])
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
          >>> R = QPolynomial(p_coeff=[1, -3, 1, -3])
          >>> b = R.mahler_separation_bound()
          >>> b
          0.111...
          
        """
        n = self.deg()

        d = float(self.discriminant())
        sep = np.sqrt(3 * np.abs(d) / n ** (n + 2)) / lin.norm(
            self.__p_coeff / self.__q_coeff, ord=2
        ) ** (n - 1)

        return sep

    def copy(self):
        p = self.__p_coeff.copy()
        q = self.__q_coeff.copy()
        res = QPolynomial(p_coeff=p, q_coeff=q, check=False)
        return res

    def squareFreeFact(self):
        r"""
        
        Examples:
          >>> R = QPolynomial(p_coeff=[1, -1, 0, 0, -1, 1])
          >>> R.squareFreeFact()
          [-1  0  0  0  1]
          [1 1 1 1 1]
          
        """
        n = self.deg()
        p = self.copy()

        while True:
            dp = p.derive()
            g = Qpolygcd(p, dp)
            if g == QPolynomial(p_coeff=[1]):
                return p

            p, r = Qpolydiv(p, g)
            if not r.isNull():
                raise AssertionError(r)

    def compose(self, S: "QPolynomial") -> "QPolynomial":
        r"""Computes self(S(X))
        
        Examples:
          >>> R = QPolynomial(p_coeff=[1, -1, 1])
          >>> S = QPolynomial(p_coeff=[-4, 5])
          >>> R.compose(S)
          [ 21 -45  25]
          [1 1 1]
          
        """
        res = QPolynomial()
        sp = QPolynomial(coeff=[Fraction(1, 1)])
        for rk in self:
            if rk != 0:
                res = res + sp * rk
            sp = sp * S

        return res

    def translate(self, a: "QPolynomial") -> "QPolynomial":
        r"""
        
        Examples:
          >>> R = QPolynomial(p_coeff=[0, 0, 1])
          >>> R.translate(2)
          [4 4 1]
          [1 1 1]
          
        """
        n = self.deg()

        q = (n + 1) * [None]
        for k in range(n + 1):
            q[k] = Fraction(0, 1)
            for p in range(k, n + 1):
                q[k] += nCr(p, k) * self[p] * a ** (p - k)

        return QPolynomial(coeff=q)

    def roots(self) -> np.array:
        r"""
        
        Examples:
          >>> R = QPolynomial(p_coeff=[-4, 0, 1])
          >>> R.roots()
          array([-2.,  2.]...
          
        """
        return P.polyroots(self.__p_coeff / self.__q_coeff)

    def newton_sum(self, d: int) -> Fraction:
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
          >>> h = QPolynomial(p_coeff=[1, 2, 3])
          >>> for d in range(5): print(h.newton_sum(d))
          2
          -2/3
          -2/9
          10/27
          -14/81
          
        """
        from AlgebraicNumber.QMatrix import QMatrix

        Dh = self.deg()

        if d == 0:
            return Fraction(numerator=Dh)

        # X is the vector of the d-th Newton's sum s_d :
        # with x_k a root of self,
        # s_d = x_1^d + x_2^d + ... + x_n^d
        # X = [s_1, s_2, ..., s_D]
        A = QMatrix.zeros(Dh, Dh)
        B = QMatrix.zeros(Dh, 1)

        tp, tq = self.getCoefficientsAsNumDen()
        tmp = tp / tq

        for r in range(Dh):
            for u in range(r + 1):
                A[r, u] = self[Dh - r + u]

            B[r, 0] = -(r + 1) * self[-r - 2]

        X = A.inv() @ B

        if d <= Dh:
            return X[d - 1, 0]

        a_lrs = QMatrix.zeros(1, Dh)
        for u in range(Dh):
            a_lrs[0, u] = -self[u] / self[-1]

        for d in range(Dh + 1, d + 1):
            sd = a_lrs @ X

            pX, qX = X.getCoefficientsAsNumDen()
            pX = np.hstack((pX[1:, 0], [sd[0, 0].numerator])).reshape((Dh, 1))
            qX = np.hstack((qX[1:, 0], [sd[0, 0].denominator])).reshape((Dh, 1))
            X = QMatrix(pX, qX)

        return sd[0, 0]

    def reverse(self, D: int = None) -> "QPolynomial":
        """This function returns the reverse polynomial associated with h, denoted rev(h)
        
        :math:`a_0 + a_1.X + a_2.X^2 + ...`

        h is represented by the sequence [a0, a1, a2, ...]

        rev(h) is defined as follows:

        :math:`rev(h) = X^{deg(h)}.h(1/X)`

        If D is > deg(h), the reversed h will be multiplied by :math:`X^k` so that the final degree is D

        Args:
          D
            The degree of the output. Must be >= deg(h)

        Returns:
          The reverse

        Examples:
          >>> h = QPolynomial(p_coeff=[1, 2, 3])
          >>> h.reverse()
          [3 2 1]
          [1 1 1]

        """
        if D is None:
            D = self.deg()

        p_rev = np.pad(
            self.__p_coeff[-1::-1],
            (D + 1 - len(self.__p_coeff), 0),
            "constant",
            constant_values=(0, 0),
        )
        q_rev = np.pad(
            self.__q_coeff[-1::-1],
            (D + 1 - len(self.__q_coeff), 0),
            "constant",
            constant_values=(0, 0),
        )

        return QPolynomial(p_coeff=p_rev, q_coeff=q_rev)

    def LogRev(self, D: int = None) -> "QPolynomial":
        """This function returns the logarithmic reverse rational power series associated with h, denoted LogRev(h)
        The result of this function is a truncature of degree D

        h is a polynomial with integer coefficients :

        :math:`a_0 + a_1.X + a_2.X^2 + ...`

        h is represented by the sequence [a0, a1, a2, ...]

        LogRev(h) is defined as follows :

        :math:`LogRev(h) = rev(h')/rev(h)`
        
        Args:
          D
            Degree of the resulting polynomial
            
        Returns:
          The logarithmic reverse
          
        Examples:
          >>> h = QPolynomial(p_coeff=[1, 2, 3])
          >>> h.LogRev()
          [ 2 -2 -2]
          [1 3 9]
          >>> h.LogRev(D=5)
          [  2  -2  -2  10 -14  -2]
          [  1   3   9  27  81 243]
          >>> h = QPolynomial(p_coeff=[0, -3])
          >>> h.LogRev()
          [1]
          [1]
 
        """
        if D is None:
            D = self.deg()

        p = np.empty(D + 1, dtype=np.int64)
        q = np.empty(D + 1, dtype=np.int64)
        for d in range(D + 1):
            s = self.newton_sum(d)
            p[d] = s.numerator
            q[d] = s.denominator

        return QPolynomial(p_coeff=p, q_coeff=q)

    def InvertLogRev(self) -> "QPolynomial":
        """Given a polynomial lr which is LogRev of h, finds back h
        
        Returns:
          The original polynomial

        Examples:
          >>> h = QPolynomial(p_coeff=[1, 2, 3])
          >>> l = h.LogRev()
          >>> l.InvertLogRev()
          [1 2 1]
          [3 3 1]
          >>> h = QPolynomial(p_coeff=[-1, 0, 1])
          >>> l = h.LogRev(D=4)
          >>> l
          [2 0 2 0 2]
          [1 1 1 1 1]
          >>> l.InvertLogRev()
          [-1  0  1]
          [1 1 1]
          >>> h = QPolynomial(p_coeff=[0, -3])
          >>> l = h.LogRev()
          >>> l.InvertLogRev()
          [0 1]
          [0 1]
          
        """
        # D is the degree of the resulting h
        D = self[0].numerator
        if self[0].denominator != 1:
            raise AssertionError(self[0])

        # Computation of (D - LogRev(h))/X as the fraction n1, d1
        p, q = self.getCoefficientsAsNumDen()
        n2 = -p[1:]
        d2 = q[1:]
        p2 = QPolynomial(p_coeff=n2, q_coeff=d2)

        # Integrate p2
        p3 = p2.integrate()

        # Exponentiate p3
        p4 = QPolynomial(p_coeff=[1])
        pdl = np.empty(D + 1, dtype=np.int64)
        qdl = np.empty(D + 1, dtype=np.int64)
        k = np.arange(D + 1, dtype=np.int64)
        coeff = np.array([factorial(x) for x in k], np.int64)
        for i in range(1, D + 1):
            # dl : DL de exp(p3[i].x^i) a l'ordre D
            s = slice(0, D + 1, i)
            nc = 1 + D // i

            pdl[:] = 0
            qdl[:] = 1

            if i <= p3.deg():
                p_p3 = p3[i].numerator
                pdl[s] = p_p3 ** k[:nc]

                q_p3 = p3[i].denominator
                qdl[s] = q_p3 ** k[:nc] * coeff[:nc]
            else:
                pdl[0] = 1

            dl = QPolynomial(p_coeff=pdl, q_coeff=qdl)
            p4 = p4 * dl

        return p4.truncate(deg=D).reverse(D=D)

    def truncate(self, deg: int) -> "QPolynomial":
        p, q = self.getCoefficientsAsNumDen()
        p = p[: deg + 1]
        q = q[: deg + 1]

        return QPolynomial(p_coeff=p, q_coeff=q)


def Qsylvester(R, S):
    """
    
    Examples:
      >>> R = QPolynomial(p_coeff=[1, -3, 1, -3, 1])
      >>> S = QPolynomial(p_coeff=[-3, 2, -9, 4])
      >>> m = Qsylvester(R, S)
      
    """
    from AlgebraicNumber.QMatrix import QMatrix

    m = R.deg()
    n = S.deg()

    p_res = np.zeros((n + m, n + m), dtype=np.int64)
    q_res = np.ones((n + m, n + m), dtype=np.int64)
    for k in range(n):
        for p in range(1 + m):
            r = R[m - p]
            p_res[k, k + p] = r.numerator
            q_res[k, k + p] = r.denominator
    for k in range(m):
        for p in range(1 + n):
            s = S[n - p]
            p_res[k + n, k + p] = s.numerator
            q_res[k + n, k + p] = s.denominator

    return QMatrix(p_res, q_res)


def Qresultant(R, S):
    """
    
    Examples:
      >>> R = QPolynomial(p_coeff=[1, -3, 1, -3, 1])
      >>> S = QPolynomial(p_coeff=[-3, 2, -9, 4])
      >>> print(Qresultant(R, S))
      -4563
      
    """
    m = Qsylvester(R, S)
    return m.det()


def Qpolydiv(n, d):
    """
    
    Examples:
      >>> n = QPolynomial(p_coeff=[-4, 0, -2, 1])
      >>> d = QPolynomial(p_coeff=[-3, 1])
      >>> q,r = Qpolydiv(n, d)
      >>> print(q)
      [3 1 1]
      [1 1 1]
      >>> print(r)
      [5]
      [1]
  
    """
    if d.isNull():
        raise ZeroDivisionError

    q = QPolynomial()
    r = n.copy()  # At each step n = d × q + r

    while not r.isNull() and r.deg() >= d.deg():
        c = r[-1] / d[-1]  # Divide the leading terms
        t = QPolynomial(coeff=(r.deg() - d.deg()) * [Fraction(0, 1)] + [c])
        q = q + t
        r = r - t * d

    return (q, r)


def Qpolygcd(a, b):
    r"""
    
    Examples:
      >>> R = QPolynomial(p_coeff=[1, -1, 0, 0, -1, 1])
      >>> S = QPolynomial(p_coeff=[-1, 0, 0, -4, 5])
      >>> g = Qpolygcd(S, R)
      >>> print(g)
      [-1  1]
      [1 1]
      
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
    res = QPolynomial(coeff=[Fraction(1, 1)])
    for q in polynomials:
        res = res * q
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
