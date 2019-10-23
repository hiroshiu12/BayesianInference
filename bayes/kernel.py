import numpy as np
import numpy.linalg as LA
import time

def gauss_kernel(x1,x2,theta1=1,theta2=2):
    """
    Return value of gauss kernel
    """
    return theta1 * np.exp( - ((x1 - x2) ** 2).sum()/theta2 )
    
def linear_kernel(x1,x2):
    """
    Return value of liear kernel
    """
    return x1@x2
    
def periodic_kernel(x1,x2,theta1=1,theta2=2):
    """
    Return value of periodic kernel
    """
    return np.exp(theta1*np.cos(((x1 - x2) ** 2).sum()/theta2))
    
def linear_gauss_kernel(x1,x2,theta1=1,theta2=2,theta3=10):
    """
    Return value of periodic kernel
    """
    return theta1 * np.exp( - ((x1 - x2) ** 2).sum()/theta2 ) + theta3*(x1@x2)