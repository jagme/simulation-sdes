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

    def simulate(self, dates):
        """
        Exact simulation of Gaussian OU process over the possibly irregular time partition dates
        e.g. {0, dt, 3dt, 4dt, ..., T}
        """
        dt = np.diff(dates)
        temp = np.exp(-self.alpha * dt)
        temp2 = self.sigma * np.sqrt((1 - temp ** 2) / (2 * self.alpha))

        X = np.zeros_like(dates)
        X[0] = self.x0

        for i in range(1, len(X)):
            X[i] = self.mu * (1 - temp[i-1]) + temp[i-1] * X[i-1] + temp2[i-1] * np.random.standard_normal()

        return X

    def conditional_mean(self, a, dt):
        return self.mu + (a - self.mu) * exp(-self.alpha * dt)

    def conditional_var(self, dt):
        return self.sigma ** 2 / 2 / self.alpha * (1 - exp(-2 * self.alpha * dt))

    def simulate_m_ou(self, dates, m):
        """
        Simulate m sample paths of the OU process
        """
        nt = np.alen(dates)
        X = np.zeros((m, nt))

        for i in range(0, m):
            X[i,] = self.simulate(dates)

        return X



# Auxiliary classes for the Background Levy driving processes of non-Gaussian OU processes

# jump size distribution


class JumpSize():

    def __init__(self, name, parameters):
        self.distribution = name
        self.parameters = parameters

    def simulate(self):
        if self.distribution == 'exponential':

            return np.random.exponential(self.parameters)
        else:
            return 0


class CPoissonProcess:
    """
    Compound Poisson process

    with initial value x0, eta inverse of intensity rate (the expected time between jumps),
    and jump distribution jump_distribution.
    """
    def __init__(self, eta, jump_distribution=JumpSize('exponential', 1), x0=0):
        self.x0 = x0
        self.eta = eta
        self.jump_distribution = jump_distribution

    def simulate(self, t_max):
        jump_sizes = [0]
        jump_times = [0]

        t = np.random.exponential(self.eta)

        while t < t_max:
            jump_times.append(t)
            t = t + np.random.exponential(self.eta)
            jump_sizes.append(self.jump_distribution.simulate())

        X = np.array(jump_times)
        Y = np.array(jump_sizes)
        return X, Y


class NonGaussianOU(SDE):
    """
    Non Gaussian OU process

    dXt = alpha*(mu - Xt)*dt + sigma*dLt, X0 = x0,

    alpha is the speed of mean reversion,
    mu is the level of mean reversion,
    sigma is the volatility and,
    Lt is the Background Driving Levy process (BDLP).
    Lt could be e.g. a Compound Poisson process with exponentially distributed jump sizes.
    """
    def __init__(self, x0, alpha, mu=0, sigma=1, bdlp=CPoissonProcess(1)):
        super(NonGaussianOU, self).__init__(x0)
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.bdlp = bdlp  # background driving Levy process

    def simulate(self, dates):
        """
        Exact simulation of Non Gaussian OU process
        """
        # jump times and jump sizes of underlying Levy process
        jump_times, jump_sizes = self.bdlp.simulate(dates[-1]) # until max. time

        # refine time partition for simulation using given dates and jump times
        grid_t = np.concatenate((dates, jump_times[1:]))
        # join all (possibly zero-size) jumps
        grid_j = np.concatenate((np.zeros_like(dates), jump_sizes[1:]))
        # and order jump times and sizes
        order_grid = np.argsort(grid_t)
        grid_t = grid_t[order_grid]
        grid_j = grid_j[order_grid]

        X = np.zeros_like(grid_t)
        X[0] = self.x0

        for i in range(1, len(X)):
            dt = grid_t[i] - grid_t[i-1]
            # for bldp a comp. Poisson process, otherwise should use infinite series representation
            X[i] = self.mu + (X[i-1] - self.mu)*exp(-self.alpha*dt) + self.sigma*grid_j[i]

        return X, grid_t, jump_times, jump_sizes


class Arithmetic2OU(SDE):
    """
    Gaussian OU + Non-Gaussian OU: Y1 + Y2
    """
    def __init__(self, x0, y1=GaussianOU(), y2=NonGaussianOU(0, 0.5)):
        super(Arithmetic2OU, self).__init__(x0)
        self.y1 = y1
        self.y2 = y2

    def simulate(self, dates):
        # simulate jump process first
        y2, grid_t, jtimes, jsizes = self.y2.simulate(dates)

        # simulate Gaussian OU over this grid to get exact solution
        y1 = self.y1.simulate(grid_t)

        x = y1 + y2

        return x, grid_t, y1, y2, jtimes, jsizes


