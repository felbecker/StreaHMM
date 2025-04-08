from distribution import Distribution



class Prior(Distribution):
    """
    The Prior class is a base class for all HMM priors, 
    where "prior" should be understood in the Bayesian sense.
    A prior is a module that takes the parameters of another module as input, defines 
    a prior distribution over them, and its outputs describe how plausible the
    parameters are. 
    """
    pass