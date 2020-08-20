import unittest
import os

import numpy as np
from numpy.testing import assert_allclose


class TestBase(unittest.TestCase):
    def get_fic_path(self, fic):
        pth = os.path.join(os.path.dirname(__file__), fic)
        return pth

    def ecrit_data_fic(self, dat, fic):
        pth = self.get_fic_path(fic)
        f = open(pth, "w")

        ns = len(dat)
        for i in range(ns):
            f.write("%f, %f\n" % (np.real(dat[i]), np.imag(dat[i])))

        f.close()

    def lit_data_fic(self, fic):
        pth = self.get_fic_path(fic)
        f = open(pth, "r")
        l = f.readlines()
        f.close()

        ns = len(l)
        ref = np.empty(ns, dtype=np.complex64)
        for i in range(ns):
            r = l[i]
            x, y = [float(v.strip()) for v in r.strip().split(",")]
            ref[i] = x + y * 1j

        return ref

    def compare_data_ref(self, dat, fic_ref):
        ref = self.lit_data_fic(fic_ref)
        err = np.max(np.abs(dat - ref))
        return err

    def assertNpArrayAlmostEqual(self, actual, desired, delta):
        """Raises an AssertionError if two objects are not equal up to desired precision.."""
        assert_allclose(
            actual, desired, atol=delta, equal_nan=False, err_msg="", verbose=True
        )

    def assertQPolynomialEqual(self, actual, desired):
        self.assertTrue(actual == desired)
