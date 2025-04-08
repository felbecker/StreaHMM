# Generate simulated data from the occasionally dishonest casino.
# The HMM has 
#  * state space {0,1} = {honest, dishonest}
#  * emission alphabet {1,..., 6}
#  * transition distribution given by matrix A
#  * emission distribution given by matrix B
 
#  Author: Mario Stanke
#          Felix Becker (generator now outputs x and y) 

import numpy as np



def make_casino_generator(sequence_length=100,
                          batch_size=2,
                          p_honest=0.8, 
                          p_dishonest=0.5, 
                          p_six=0.5,
                          p_start_honest=1.0):
    """
    Creates a casino generator with the given parameters.
    The generator yields a tuple (x,y) where x is the observed sequence
    and y is the hidden state sequence.
    Args:
        sequence_length (int): Length of the sequences to generate.
        batch_size (int): Number of sequences to generate in each batch.
        p_honest (float): P(honest | honest)
        p_dishonest (float): P(dishonest | dishonest)
        p_six (float): P(6 | dishonest)
        p_start_honest (float): Probability of starting in the honest state.
    """
    s,n = 6, 2
    alphabet = list(range(1,s+1))
    states = list(range(n))
    A = np.array([[p_honest, 1-p_honest], [1-p_dishonest, p_dishonest]])
    B = np.array([[1./6]*6, [(1.-p_six)/5.]*5 + [p_six]])

    def casino_generator():

        x = np.zeros((batch_size, sequence_length, s), dtype=np.float32)
        y = np.zeros((batch_size, sequence_length, n), dtype=np.float32)

        for j in range(batch_size):
            q = np.random.choice(states, p=[p_start_honest, 1-p_start_honest])
            c = np.random.choice(alphabet, p=B[q])
            x[j, 0, c-1] = 1    
            y[j, 0, q] = 1
            for i in range(1, sequence_length):
                q = np.random.choice(states, p=A[q])
                c = np.random.choice(alphabet, p=B[q])
                x[j, i, c-1] = 1    
                y[j, i, q] = 1
        yield x,y
    return casino_generator