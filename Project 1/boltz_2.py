import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from ast2000tools.constants import k_B, m_H2


#* CONSTANTS
N = 1e5
T = 3000

sigma = np.sqrt(k_B*T/m_H2)


def p(vx, sigma):
    return vx*4*np.pi*vx**2/((2*np.pi*sigma**2)**(3/2))*np.exp(-1/2*(vx/sigma)**2)

def P(f, a, b,  sigma):
    I = sp.integrate.quad(f, a, b, args=(sigma))
    
    return I[0]


#print(P(p, 0, np.inf, sigma))


omega = (m_H2/(2*np.pi*k_B*T))**(3/2)
print(omega*sigma**2*2*4*np.pi)