import numpy as np
import numpy.linalg as LA

class gauss_mix_gibbs():

    def __init__(self, k_num, d_num):

        self.d_num = d_num
        self.k_num = k_num
        # Parameters for prior distribution
        self.init_alpha = np.ones(k_num)
        self.init_m = np.random.randn((d_num))
        self.init_beta = 1
        self.init_nu = d_num
        self.init_w = np.eye(self.d_num)

    def gibbs_sample(self, X, max_iter=100):

        print('init_m', self.init_m)
        # Container of results
        samp_z = np.empty((max_iter, X.shape[0], self.k_num))
        samp_mu = np.empty((max_iter, self.k_num, self.d_num))
        samp_lam = np.empty((max_iter, self.k_num, self.d_num, self.d_num))
        samp_pi = np.empty((max_iter, self.k_num))

        # Start MCMC with p(z | mu, lambda, pi, X)
        # Initialize parameter for mu, lambda, pi
        mu = np.array([[0, 0], [1, 1]])
        lam = np.array([np.eye(2), np.eye(2)])
        pi = np.array([0.5, 0.5])

        for i in range(max_iter):

            # Sample p(z | mu, lambda, pi, X)
            if i != 0:
                eta_pre = np.array([
                    -0.5 * (X[n] - samp_mu[i - 1, k]).reshape(1, -1) @
                    samp_lam[i - 1, k] @ (X[n] - samp_mu[i - 1, k]).reshape(-1,
                                                                            1) +
                    0.5 * np.log(LA.det(samp_lam[i - 1, k, :, :])) +
                    np.log(samp_pi[i - 1, k])
                    for n in range(X.shape[0])
                    for k in range(self.k_num)
                ]).reshape(X.shape[0], self.k_num)
            else:
                eta_pre = np.array([
                    -0.5 * (X[n] - mu[k]).reshape(1, -1) @ lam[k] @ (
                                X[n] - mu[k]).reshape(-1, 1) +
                    0.5 * np.log(LA.det(lam[k])) +
                    np.log(pi[k])
                    for n in range(X.shape[0])
                    for k in range(self.k_num)
                ]).reshape(X.shape[0], self.k_num)

            const = log_sum_exp(eta_pre)
            eta = np.exp(eta_pre - const)

            samp_z[i] = np.array(
                [multinomial.rvs(n=1, p=eta_n) for eta_n in eta])

            sum_part = np.array([
                samp_z[i, n, k] * X[n].reshape(-1, 1) @ X[n].reshape(1, -1)
                for k in range(self.k_num)
                for n in range(X.shape[0])
            ]).reshape(self.k_num, X.shape[0], self.d_num, self.d_num).sum(
                axis=1)

            beta_hat = np.array([
                samp_z[i, :, k].sum() + self.init_beta
                for k in range(self.k_num)
            ])

            sum_m_hat = np.array([
                samp_z[i, n, k] * X[n]
                for k in range(self.k_num)
                for n in range(X.shape[0])
            ]).reshape(self.k_num, X.shape[0], self.d_num).sum(axis=1)

            m_hat = np.array([
                (sum_m_hat[k] + self.init_beta * self.init_m) / beta_hat[k]
                for k
                in range(self.k_num)])

            w_inv = np.array([
                sum_part[k] +
                self.init_beta * self.init_m.reshape(-1,
                                                     1) @ self.init_m.reshape(
                    1, -1) -
                beta_hat[k] * m_hat[k].reshape(-1, 1) @ m_hat[k].reshape(1,
                                                                         -1) +
                LA.inv(self.init_w)
                for k in range(self.k_num)
            ])

            nu = np.array([samp_z[i, :, k].sum() + self.init_nu for k in
                           range(self.k_num)])
            #             print(w)
            samp_lam[i] = np.array([
                wishart.rvs(df=nu[k], scale=LA.inv(w_inv[k]))
                for k in range(self.k_num)
            ])
            #             print(m_hat.shape)
            samp_mu[i] = np.array([
                multivariate_normal.rvs(mean=m_hat[k], cov=LA.inv(
                    beta_hat[k] * samp_lam[i, k]))
                for k in range(self.k_num)
            ])

            alpha_hat = np.array([
                samp_z[i, :, k].sum() + self.init_alpha[k]
                for k in range(self.k_num)
            ]).ravel()
            #             print(alpha_hat.shape)
            samp_pi[i] = dirichlet.rvs(alpha=alpha_hat)

        return samp_z, samp_mu, samp_lam, samp_pi