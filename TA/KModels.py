from sympy import *
import numpy as np



# Here I define the "raw" versions of the kinetic models. This is how a model is defined the first time that it is created.

def rawFloppyRotor():
    k1, k2, knr, kr = symbols('k1 k2 knr kr')       # Bate constants
    time = symbols('time')                       # Time
    A,B = symbols('A,B') # Populations (S1, T1, and emitted Photons)
    KineticModel = Matrix([
           [ -(k1+kr) ,  +k2    ],   # d[M]
           [ +k1   ,   -(knr+k2)]    # d[B]
                                 ])   
    C0 = Matrix([A,B]) # Initial population
    EigenVectors, EigenValues = KineticModel.diagonalize()
    Ct = EigenVectors@exp(EigenValues*time)@(EigenVectors**-1)@C0
    Ct = lambdify((time,k1,k2,knr,kr,A,B),Ct,'numpy')
    return Ct

def rawSequential():
    k1, knr, kr = symbols('k1 knr kr')       # Bate constants
    time = symbols('time')                       # Time
    A,B = symbols('A,B') # Populations (S1, T1, and emitted Photons)
    KineticModel = Matrix([
           [ -(k1+kr) ,    0   ],   # d[M]
           [ +k1   ,   -(knr)]    # d[B]
                                 ])   
    C0 = Matrix([A,B]) # Initial population
    EigenVectors, EigenValues = KineticModel.diagonalize()
    Ct = EigenVectors@exp(EigenValues*time)@(EigenVectors**-1)@C0
    Ct = lambdify((time,k1,knr,kr,A,B),Ct,'numpy')
    return Ct










# Below we can load the solutions directly. Much faster!



#def Sequential():
Sequential = lambda time, k1, kr, knr,  A, B: np.array([[A*np.exp(time*(-k1-kr))],[A*(k1*np.exp(time*(-k1 -kr))/(-k1 + knr - kr) - k1*np.exp(-knr*time)/(-k1 + knr - kr)) + B*np.exp(-knr*time)]])
   # return Ct





#def FloppyRotor():
FloppyRotor = lambda time, k1, kr, knr, k2, A, B: np.array([
    [A*(2.0*k2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))) - 2*k2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))) + B*(4*k2**2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))*(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))) - 4*k2**2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))*(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))))],
    [A*(-np.exp(time*(-k1/2.0 - k2/2 - knr/2 - kr/2 - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/(2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))) + np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/(2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))) + B*(2*k2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))) - 2*k2*np.exp(time*(-k1/2 - k2/2 - knr/2 - kr/2 + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)/2))/((2*k2/(k1 - k2 - knr + kr + np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)) - 2*k2/(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2)))*(k1 - k2 - knr + kr - np.sqrt(k1**2 + 2*k1*k2 - 2*k1*knr + 2*k1*kr + k2**2 + 2*k2*knr - 2*k2*kr + knr**2 - 2*knr*kr + kr**2))))]])
    #return Ct 
