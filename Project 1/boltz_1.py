import numpy as np 
import scipy as sp 



def f(x:float, mu:float, sigma:float) -> float:
    frac = 1/(sigma*np.sqrt(2*np.pi))
    x_exp = -1/2*((x-mu)/sigma)**2
    
    return frac*np.exp(x_exp)


def P(a, b,  mu, sigma):
    I = sp.integrate.quad(f, a, b, args=(mu, sigma))
    
    return I[0]



if __name__ == '__main__':
    
    
    mu = 0
    sigma = 0.2
    
    print(P(-1*sigma, sigma, mu, sigma))
    print(P(-2*sigma, 2*sigma, mu, sigma))
    print(P(-3*sigma, 3*sigma, mu, sigma))
    
    