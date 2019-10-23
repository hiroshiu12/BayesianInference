import numpy as np
import numpy.linalg as LA
import time

class gp_classifier():
    def __init__(self,kernel_func):
        """
        Assumed class attribute
        fit():
        self.C_N
        """
        self.kernel_func = kernel_func
        
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
            self.kernel_func(X[i],X[j])
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
            if i +1 == max_iter:
                self.pos_mean = a_N_new
        # Compute marginal likelihood
        W_n = np.diag(
            self._sigmoid(self.pos_mean) * (1 - self._sigmoid(self.pos_mean)))
        lnp_a_asta = -(0.5 * a_N.reshape(1,-1)@LA.inv(self.C_N)@a_N.reshape(-1,1)) \
                     -(0.5 * self.y.shape[0]*np.log(2*np.pi))
#                      -(0.5 * np.log(LA.det(self.C_N)))
#         print('determinant of C_N',LA.det(self.C_N))
#         print('First term', -(0.5 * a_N.reshape(1,-1)@LA.inv(self.C_N)@a_N.reshape(-1,1))[0][0])
#         print('Second term',-(0.5 * self.y.shape[0]*np.log(2*np.pi)))
#         print('Third term',-(0.5 * np.log(LA.det(self.C_N))))
#         print('Third term',-(0.5 * np.log(LA.slogdet(self.C_N)[1])))

#         print('The value of lnp_a_asta :',lnp_a_asta)
        lnp_p_tn_a_asta = self.y.reshape(1,-1)@a_N.reshape(-1,1) \
                        -np.log(1+np.exp(a_N)).sum()
#         print('The value of lnp_p_tn_a_asta : ',lnp_p_tn_a_asta)
        ln_wn_cn = 0.5 * (LA.slogdet(W_n + LA.inv(self.C_N))[1])
#         print('The ln_wn_cn : ',ln_wn_cn)
        self.mlh = (lnp_a_asta + lnp_p_tn_a_asta - ln_wn_cn + \
                   (0.5/self.y.shape[0] * np.log(2 * np.pi)))[0][0]
            
    def predict_proba(self,X):
        """
        parameter:
        X : design matrix of prediction
        """
        k = np.array([
                self.kernel_func(X[i], self.design_matrix[j])
                for i in range(X.shape[0])
                for j in range(self.design_matrix.shape[0])
            ]).reshape(X.shape[0], self.design_matrix.shape[0])
#         k = self.kernel_func(
#             np.tile(self.design_matrix,(1,X.shape[0],1)).reshape(X.shape[0],self.design_matrix.shape[0],X.shape[1]),
#                 np.tile(X,(1,1,self.design_matrix.shape[0])).reshape(X.shape[0],self.design_matrix.shape[0],X.shape[1])
#             )
#         print('elapsed time : ', t2-t1)
        W_n = np.diag(self._sigmoid(self.pos_mean)*(1-self._sigmoid(self.pos_mean)))
        # Compute mean and cov of p(a_{N+1} | t_N)
        temp_mean = k@(self.y - self._sigmoid(self.pos_mean)).reshape(-1,1).ravel()
        var_X = np.array([
            self.kernel_func(X[i],X[i])
            for i in range(X.shape[0])
        ])
        temp_cov = var_X - \
                                    np.diag(k@LA.inv((LA.inv(W_n) + self.C_N))@k.T)
        prob = self._sigmoid(temp_mean / np.sqrt((1 + np.pi*(temp_cov**2))/8)) 
        
        return np.vstack([1-prob,prob]).T
    
    def get_marginal_likelihood(self):
        
        return self.mlh
        
    def _sigmoid(self,x):
        """
        Return the value mapped by sigmoid function
        """
        return 1 / (1+np.exp(-x))
