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

    with initial value x0, eta intensity function,
    and jump distribution jump_distribution.
    If eta is constant, it is interpreted as the mean arrival rate
    """
    def __init__(self, eta, jump_distribution=JumpSize('exponential', 1), x0=0):
        self.x0 = x0
        self.eta = eta           # scale parameter, inverse of rate parameter. It is such mean is eta
        self.jump_distribution = jump_distribution

    def simulate(self, tmax):
        jumps = [0]
        jump_times = [0]

        t = np.random.exponential(self.eta)

        while t < tmax:
            jump_times.append(t)
            t = t + np.random.exponential(self.eta)
            jumps.append(self.jump_distribution.simulate())


        X = np.array(jump_times)
        Y = np.array(jumps)
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

    def simulate(self, nt, t_max):
        # jump times and jump sizes of underlying Levy process
        jump_times, jump_sizes = self.bdlp.simulate(t_max)

        grid0 = np.linspace(0, t_max, nt + 1)
        grid_t = np.concatenate((grid0, jump_times[1:]))
        grid_j = np.concatenate((np.zeros_like(grid0), jump_sizes[1:]))

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
