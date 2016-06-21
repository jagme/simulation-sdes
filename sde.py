import numpy as np

from math import exp


class SDE:
    """
    Class for stochastic differential equations (SDEs)
    """

    def __init__(self, x0):
        self.x0 = x0


class GaussianOU(SDE):
    """
    Class for Gaussian Ornstein-Uhlenbeck processes (aka Vasiceck model)

        dXt = alpha*(mu - Xt)*dt + sigma*dW(t), X0 = x0,

    where W(t) is a standard Wiener process,
    alpha is the speed of mean reversion,
    mu is the level of mean reversion, and
    sigma is the volatility.
    """

    def __init__(self, x0=0, mu=0, alpha=1, sigma=1):
        super().__init__(x0)
        self.mu = mu
        self.alpha = alpha
        self.sigma = sigma

    def get_parameters(self):
        return self.mu, self.alpha, self.sigma


    def simulate(self, nt, t_max):
        dt = t_max / nt
        temp = np.exp(-self.alpha * dt)
        X = np.zeros(nt + 1)
        X[0] = self.x0
        temp2 = self.sigma * np.sqrt((1 - temp ** 2) / (2 * self.alpha))
        for i in range(1, nt + 1):
            X[i] = self.mu * (1 - temp) + temp * X[i - 1] + temp2 * np.random.standard_normal()

        return X

    def conditional_mean(self, a, dt):
        return self.mu + (a - self.mu) * exp(-self.alpha * dt)

    def conditional_var(self, dt):
        return self.sigma ** 2 / 2 / self.alpha * (1 - exp(-2 * self.alpha * dt))


    def simulate_m_ou(self, nt, t_max, m):
        """
        Simulate m sample paths of the OU process
        """
        X = np.zeros((m, nt))
        # X = [self.simulate(nt, t_max) for i in range(m + 1)]
        for i in range(0, m):
            X[i,] = self.simulate(nt, t_max)

        return X



