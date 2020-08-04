import numpy as np
from mpmath import pslq

from AlgebraicNumber.AlgebraicNumber import AlgebraicNumber
from AlgebraicNumber.utils import cpslq
from AlgebraicNumber.apslq import apslq
from AlgebraicNumber.identification import pslq

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestPSLQ (TestBase):
    def ntest_apslq(self):
        vec = apslq([np.log(2),np.log(3),np.log(4),np.log(6)], D=0)
        ref = np.array([1,1,0,-1])
        if vec[0] < 0:
            ref *= -1
            
        self.assertNpArrayAlmostEqual(vec,ref,delta=1.e-8)
        
    def ntest_pslq_cplxe(self):
        z0 = -2 + 3*1j
        vec = pslq([1, z0, z0**2, z0**3, z0**4])
        ref = np.array([0, 0, 13, 4, 1])
        if vec[0] < 0:
            ref *= -1
            
        self.assertNpArrayAlmostEqual(vec,ref,delta=1.e-8)
        
    def test_pslq_log(self):
        vec = pslq([np.log(2),np.log(3),np.log(4),np.log(6)])
        ref = np.array([1,1,0,-1])
        if vec[0] < 0:
            ref *= -1
            
        self.assertNpArrayAlmostEqual(vec,ref,delta=1.e-8)
        
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

if __name__ == '__main__':
    unittest.main()
    
    