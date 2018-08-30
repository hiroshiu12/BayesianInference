import numpy as np
import numpy.linalg as LA

class linear_bayes:
    """
    Representation of Linear Regression
    with bayesian inference
    """
    def __init__(self,lam,pre,mu,num_dim=3):
        """
        lam : parameter of variance for norm which yield y
        pre : parameter of precision-matrix for multivariate
        normal distribution
        mu : parameter for multivariate normal dist
        num_dim : dimention of linear regression
        """
        self.lam = lam
        self.pre = pre
        # tranform into vector style
        mu = mu.reshape(-1,1)
        self.mu = mu
        self.num_dim = num_dim

    def create_input(self,x):
        """
        x is expected (1,n) numpy.
        """
        # If x is given as 1-dimentional array.
        # trainsfom it into vector.
        try:
            x.shape[1]
        except IndexError:
            x = x.reshape((1,x.shape[0]))

        # Whatever number you gave, intercept is gonna be 1
        intercept = np.ones((1,x.shape[1]))
        x_vect = np.concatenate([intercept,x],axis=0)

        for i in range(self.num_dim - 2):
            x_vect = np.concatenate([x_vect,x ** (i+2)],axis=0)

        return x_vect

    def post_dist(self,x,y):
        """
        Return : precision matrix, mean vector
        """
        # store x,y so as to compute marginal likelihood
        self.x = x
        self.y = y
        
        x = self.create_input(x)
        self.pos_pre = self.lam * (x@x.T) + self.pre
        self.pos_mean = LA.inv(self.pos_pre)@((self.lam * ((y*x).sum(axis=1).reshape(x.shape[0],-1)) + self.pre @ self.mu))

        return self.pos_pre, self.pos_mean
    
    def marginal_dist(self):
        """
        Computer "marginal distribution" or "evidence" of p(Y|X). 
        """
        
        evidence = (- 1/2) * ((self.lam * (y **2).sum(0)) - np.log(self.lam) + np.log(2*np.pi) + self.mu.T@self.pre@self.mu - 
                              np.log(LA.det(self.pre)) -  self.pos_mean.T@self.pos_pre@self.pos_mean + np.log(LA.det(self.pos_pre)))
        return evidence
        
    def predict(self,x):
        """
        Return prediction
        """
        x = self.create_input(x)

        mu_asta = self.pos_mean.T @ x
        lam_asta = (1/self.lam) + np.diag(x.T@LA.inv(self.pos_pre)@x)

        return mu_asta.ravel(),lam_asta.ravel()
