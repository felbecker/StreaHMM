from module import Module
from utility import Tensor



class Distribution(Module):
    """
    Base class for all probabilistic modules in the HMM framework. Defines basic functionality that is
    common to all distributions, such as probability density functions, sampling, and expected values.
    """
    def __init__(self):
        pass


    def prob(self, x : Tensor) -> Tensor:
        """
        Returns the probability or density of the given observation x.
        """
        pass


    def log_prob(self, x : Tensor) -> Tensor:
        """
        Returns the log probability or density of the given observation x.
        """
        pass


    def mean(self) -> Tensor:
        """
        Returns the expected value of the distribution, if applicable.
        """
        pass


    def sample(self) -> Tensor:
        """
        Returns a sample from the distribution, if applicable.
        """
        pass