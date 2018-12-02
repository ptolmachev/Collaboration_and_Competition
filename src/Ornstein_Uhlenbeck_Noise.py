import numpy as np


class Ornstein_Uhlenbeck_Noise():
    def __init__(self, dimensions, mu = 0, sigma = 0.2, theta = 0.15):
        self.dimensions = dimensions
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        if type(self.dimensions) == tuple:
            self.state = self.mu+self.sigma*np.random.randn(*self.dimensions)
        else:
            self.state = self.mu+self.sigma*np.random.randn(self.dimensions)

    def generate_noise(self):
        if type(self.dimensions) == tuple:
            self.state = self.state + (self.mu*np.ones(self.dimensions) - self.state)*self.theta + self.sigma*np.random.randn(*self.dimensions)
        else:
            self.state = self.state + (self.mu - self.state)*self.theta + self.sigma*np.random.randn(self.dimensions)
        return self.state


 # Quick unit test
# Noise = Ornstein_Uhlenbeck_Noise((2,2))
# vals = []
# for i in range(100):
#     vals.append(Noise.generate_noise()[0])
#
# from matplotlib import pyplot as plt
# plt.plot(vals)
# plt.show()