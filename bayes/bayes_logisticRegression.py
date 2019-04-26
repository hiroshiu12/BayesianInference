import numpy as np
import numpy.linalg as LA

class bayes_logisticRegression():
    def __init__(self, m, pre):
        """
        m :  mean of prior distribution of w
        cov : covariance matrix of prior distribution
        """
        self.m = m.reshape(-1, 1)
        self.pre = pre
        # Initial value for newton raphson  method
        self.w_mod = np.zeros(m.shape)
        self.hessian = np.eye(m.shape[0])

    def fit(self, X, y, eps=0.0001, max_iter=10000, meth='laplace'):
        """
        Compute posterior distribution
        """
        y = y.reshape(-1, 1)
        # Start newton raphson method to get mod of posterior
        for _ in range(max_iter):
            # Compute derivative of posterior distribution
            derivative = X.T @ (
                    y - self._sigmoid(X @ self.w_mod)) - (
                                     self.pre @ (self.w_mod - self.m)
                                     )

            diag = np.array([
                self._sigmoid(self.w_mod.T @ x.reshape(-1, 1)) * (1 - (
                    self._sigmoid(self.w_mod.T @ x.reshape(-1, 1))))
                for x
                in X
            ]).ravel()

            hessian = - (X.T @ np.diag(diag) @ X) - self.pre
            w_new = self.w_mod - (LA.inv(hessian) @ derivative)
            if abs(w_new - self.w_mod).mean() <= eps:
                break

            self.w_mod = w_new.copy()
        #             print('w_mod :',self.w_mod.shape)

        self.w_mod = w_new.copy()
        self.hessian = hessian.copy()

        # Compute model evidence

    def predict_proba_map(self, X):
        """
        Compute sigma(wT @ X) with expectation of w.
        """
        return self._sigmoid(X @ self.w_mod)

    def predict_proba(self, X):
        """
        Compute sigma(wT @ X) with expectation of w.
        """
        nume = self.w_mod.T @ X
        denomi = (1 + ((np.pi * (X.T @ LA.inv(-self.hessian) @ X)) / 8)) ** (
                    1 / 2)

        return self._sigmoid(nume / denomi)

    def evidence(self, X):
        """
        Compute model evidence
        """

        pass

    def _sigmoid(self, X):
        """
        Compute value of sigmoid function
        """
        return 1 / (1 + (np.e ** (-X)))