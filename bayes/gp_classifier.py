import numpy as np
import numpy.linalg as LA

class gp_classifier():
    def __init__(self,):
        """
        Assumed class attribute
        fit():
        self.C_N
        """
        
    def fit(self,X,y,eps=1e-3,max_iter=10000):
        """
        parameter:
        X : design matrix
        y : label
        max_iter : maximum number of iteration for newton
        eps : criteria for newton raphson
        """
        # Store design matrix for prediction
        self.design_matrix = X
        self.y = y
        # Compute gram matrix and inverse of it
        self.C_N = np.array([
            self._gauss_kernel(X[i],X[j])
            for i in range(X.shape[0])
            for j in range(X.shape[0])
        ]).reshape(X.shape[0],X.shape[0])
        # Compute mod of p(a_N|t_N)
        # initialize a_N
        a_N = np.zeros(X.shape[0])
        for i in range(max_iter):
            W_n = np.diag(self._sigmoid(a_N)*(1-self._sigmoid(a_N)))
            a_N_new = self.C_N @ LA.inv(np.eye(X.shape[0]) +\
                                   W_n@self.C_N)@(y-self._sigmoid(a_N) + W_n@a_N)
            
            if abs(a_N_new - a_N).sum() < eps:
                W_n = np.diag(self._sigmoid(a_N_new)*(1-self._sigmoid(a_N_new)))
                # "posterior" means p(a_{N} | t_N).
                # Approximated by laplace approximation
                self.pos_mean = a_N_new
                break
            a_N = a_N_new
            
    def predict_proba(self,X):
        """
        parameter:
        X : design matrix of prediction
        """
        k = np.array([
                self._gauss_kernel(X[i], self.design_matrix[j])
                for i in range(X.shape[0])
                for j in range(self.design_matrix.shape[0])
            ]).reshape(X.shape[0], self.design_matrix.shape[0])
        
        W_n = np.diag(self._sigmoid(self.pos_mean)*(1-self._sigmoid(self.pos_mean)))
        # Compute mean and cov of p(a_{N+1} | t_N)
        temp_mean = k@(self.y - self._sigmoid(self.pos_mean)).reshape(-1,1).ravel()
        var_X = np.array([
            self._gauss_kernel(X[i],X[i])
            for i in range(X.shape[0])
        ])
        temp_cov = var_X - \
                                    np.diag(k@LA.inv((LA.inv(W_n) + self.C_N))@k.T)
        prob = self._sigmoid(temp_mean / np.sqrt((1 + np.pi*(temp_cov**2))/8)) 
        
        return np.vstack([1-prob,prob]).T
    
    def _gauss_kernel(self, x1,x2,theta1=1,theta2=1):
        """
        Return value of gauss kernel
        """
        return theta1 * np.exp( - ((x1 - x2) ** 2).sum()/theta2 )
    
    def _sigmoid(self,x):
        """
        Return the value mapped by sigmoid function
        """
        return 1 / (1+np.exp(-x))
