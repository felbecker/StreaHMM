from distribution import Distribution
from utility import Tensor
from enum import Enum
import viterbi
import forward_backward



class ScanStrategy(Enum):
    """
    Defines how the scan operation is performed on sequences.
    LINEAR: A linear scan strategy that processes the sequence by iterating over each time step one by one.
    PARALLEL: A parallel scan strategy that trades off memory for speed via parallelization.
    """
    LINEAR = 1,
    PARALLEL = 2


class Hmm(Distribution):
    """
    A modular hidden Markov model (HMM).
    """

    def forward_log_prob(self, 
                          x : Tensor, 
                          x_global : Tensor = None, 
                          scan_strategy : ScanStrategy = ScanStrategy.LINEAR) -> Tensor:
        """
        Runs the vectorized and differentiable forward algorithm and returns the forward log-probabilities.
        Args:
            x (Tensor or list of Tensors): The input sequence of observations. When using multiple emitters, 
                                        this should be a list of Tensors, one for each emitter.
                                        The tensors should be of shape (B, L, D), (1, B, L, D) or (M, B, L, D), 
                                        where M is the number of models, B is the batch size, 
                                        L is the sequence length, and D is the feature dimension.
            x_global (Tensor): A global input tensor of shape (B, D) or (M, B, D) that has no time dimension
                                and is used by some emitters and transitioners.
            scan_strategy (ScanStrategy): The scan strategy to use for the forward algorithm. See ScanStrategy for details.
        Returns:
            Tensor: The forward log-probabilities for the input sequence of shape (M, B, L, Q), 
                    where Q is the maximum number of states in all HMMs.
        """
        pass


    def backward_log_prob(self, 
                           x : Tensor, 
                           x_global : Tensor = None, 
                           scan_strategy : ScanStrategy = ScanStrategy.LINEAR) -> Tensor:
        """
        Runs the vectorized and differentiable backward algorithm and returns the backward log-probabilities.
        Args:
            x (Tensor or list of Tensors): The input sequence of observations. When using multiple emitters, 
                                        this should be a list of Tensors, one for each emitter.
                                        The tensors should be of shape (B, L, D), (1, B, L, D) or (M, B, L, D), 
                                        where M is the number of models, B is the batch size, 
                                        L is the sequence length, and D is the feature dimension.
            x_global (Tensor): A global input tensor of shape (B, D) or (M, B, D) that has no time dimension
                                and is used by some emitters and transitioners.
            scan_strategy (ScanStrategy): The scan strategy to use for the forward algorithm. See ScanStrategy for details.
        Returns:
            Tensor: The backward log-probabilities for the input sequence of shape (M, B, L, Q), 
                    where Q is the maximum number of states in all HMMs.
        """
        pass


    def viterbi_algorithm(self, 
                          x : Tensor, 
                          x_global : Tensor = None, 
                          scan_strategy : ScanStrategy = ScanStrategy.LINEAR) -> Tensor:
        """ 
        Runs the vectorized Viterbi algorithm and returns the most likely state sequence.
        Args:
            x (Tensor): The input sequence of observations. 
                        The tensor should be of shape (B, L, D), (1, B, L, D) or (M, B, L, D), 
                        where M is the number of models, B is the batch size, 
                        L is the sequence length, and D is the feature dimension.
            x_global (Tensor): A global input tensor of shape (B, D) or (M, B, D) that has no time dimension
                                and is used by some emitters and transitioners.
            scan_strategy (ScanStrategy): The scan strategy to use for the forward algorithm. See ScanStrategy for details.
        Returns:
            Tensor: The most likely state sequence for the input sequence of shape (M, B, L), 
                    where M is the number of models and B is the batch size.
                    The tensor contains the indices of the most likely states for each time step.
        """
        pass


    def posterior_state_log_prob(self, 
                                 x : Tensor, 
                                 x_global : Tensor = None, 
                                 scan_strategy : ScanStrategy = ScanStrategy.LINEAR) -> Tensor:
        """ 
        Runs the vectorized and differentiable forward-backward algorithm and returns the posterior state probabilities,
        i.e. the probability of being in a certain state at a certain time step given the entire sequence.
        Args:
            x (Tensor): The input sequence of observations. 
                        The tensor should be of shape (B, L, D), (1, B, L, D) or (M, B, L, D), 
                        where M is the number of models, B is the batch size, 
                        L is the sequence length, and D is the feature dimension.
            x_global (Tensor): A global input tensor of shape (B, D) or (M, B, D) that has no time dimension
                                and is used by some emitters and transitioners.
            scan_strategy (ScanStrategy): The scan strategy to use for the forward algorithm. See ScanStrategy for details.
        Returns:
            Tensor: The most likely state sequence for the input sequence of shape (M, B, L), 
                    where M is the number of models and B is the batch size.
                    The tensor contains the indices of the most likely states for each time step.
        """
        pass


    def _scan(self, 
              x : Tensor, callable, scan_strategy : ScanStrategy = ScanStrategy.LINEAR) -> Tensor:
        """
        Applies a callable to each element of the input tensor x.
        """
        pass