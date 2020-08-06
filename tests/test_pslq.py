import numpy as np
import scipy.linalg as lin

from AlgebraicNumber.AlgebraicNumber import AlgebraicNumber
from AlgebraicNumber.utils import cpslq
from AlgebraicNumber.apslq import apslq
from AlgebraicNumber.identification import pslq
from AlgebraicNumber.spslq import *

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestPSLQ(TestBase):
    def test_hyperplane_matrix(self):
        z0 = -2 + 3 * 1j
        v = np.array([1, z0, z0 ** 2, z0 ** 3, z0 ** 4])
        n = len(v)
        t = 2
        x = np.empty((n, t))
        x[:, 0] = np.real(v)
        x[:, 1] = np.imag(v)

        x2, b2, H = hyperplane_matrix(x)
        P = H @ H.T

        W = np.hstack((x2, H))

        # Orthogonalite de W
        self.assertAlmostEqual(lin.norm(W.T @ W - np.eye(n)), 0, delta=1e-8)

        self.assertAlmostEqual(lin.norm(H, ord="fro"), np.sqrt(n - t), delta=1e-8)

        # Orthogonalite de la matrice
        self.assertAlmostEqual(lin.norm(H.T @ H - np.eye(H.shape[1])), 0, delta=1e-8)

        # x dans l'hyperplan defini par la matrice H
        self.assertAlmostEqual(lin.norm(x.T @ H), 0, delta=1e-8)

        self.assertAlmostEqual(lin.norm(P, ord="fro"), np.sqrt(n - t), delta=1e-8)

        P2 = np.eye(n)
        for i in range(t):
            P2 -= np.outer(x2[:, i], x2[:, i])
        self.assertAlmostEqual(lin.norm(P - P2, ord="fro"), 0, delta=1e-8)

        self.assertAlmostEqual(lin.norm(P - P @ P, ord="fro"), 0, delta=1e-8)

    def test_generalized_hermite_reduction(self):
        z0 = -2.1 + 3.2 * 1j
        v = np.array([1, z0, z0 ** 2, z0 ** 3, z0 ** 4])
        n = len(v)
        t = 2
        x = np.empty((n, t))
        x[:, 0] = np.real(v)
        x[:, 1] = np.imag(v)

        x2, b2, H = hyperplane_matrix(x)
        D = generalized_hermite_reduction(H)

        self.assertAlmostEqual(lin.det(D), 1, delta=1e-8)

        H2 = D @ H
        for i in range(n - t):
            self.assertAlmostEqual(np.abs(H[i, i]), np.abs(H2[i, i]), delta=1e-8)
            for k in range(i + 1, n):
                self.assertTrue(np.abs(H2[k, i]) <= np.abs(H2[i, i]) / 2)

    def ntest_spslq_log(self):
        x = np.empty((4, 1))
        x[:, 0] = [np.log(2), np.log(3), np.log(4), np.log(6)]

        vec = spslq(x)
        ref = np.array([1, 1, 0, -1])
        if vec[0] < 0:
            ref *= -1

        self.assertNpArrayAlmostEqual(vec, ref, delta=1.0e-8)

    def ntest_spslq_cplxe(self):
        z0 = -2 + 3 * 1j
        v = np.array([1, z0, z0 ** 2, z0 ** 3, z0 ** 4])
        n = len(v)
        t = 2
        x = np.empty((n, t))
        x[:, 0] = np.real(v)
        x[:, 1] = np.imag(v)

        vec = spslq(x)
        ref = np.array([1, 1, 0, -1])
        if vec[0] < 0:
            ref *= -1

        self.assertNpArrayAlmostEqual(vec, ref, delta=1.0e-8)


# vec = cpslq([np.log(2),np.log(3),np.log(4),np.log(6)])
# vec = pslq([np.log(2),np.log(3),np.log(4),np.log(6)])

# print(vec)

# z0 = -2 + 3*1j
# cpslq([1,z0,z0**2,z0**3, z0**4])

# z = AlgebraicNumber.constant(-2 + 3*1j)
# print('-2+3.i', z.coeff)

# z = AlgebraicNumber.unity() + AlgebraicNumber.imaginary()
# print('1+i', z.coeff)

# z = AlgebraicNumber([2,-2,1], 1+1j, _nosimp=True)
# zc = AlgebraicNumber([2,-2,1], 1-1j, _nosimp=True)
# # zc = z.conj()

# print(72*'=')
# m = z*zc
# print(m.coeff)

if __name__ == "__main__":
    unittest.main()
