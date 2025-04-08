# Generate simulated data from a simplified gene prediction model.
# The HMM has 
#  * state space {0,1,2,3} = {IR, E0, E1, E2}
#  * emission alphabet {0,1,2,3} = {A,C,G,T}
#  * transition distribution given by matrix A
#  * emission distribution given by matrix B
#  * always starts in state IR
#
#  IR -> E0 -> E1 -> E2
#  ^------^----------|        
#
#  Author: Felix Becker

import numpy as np



def make_gene_structure_generator(sequence_length=100,
                                    batch_size=2,
                                    p_enter_E0=0.1,
                                    p_enter_IR=0.2,
                                    IR_nuc=[0.25, 0.25, 0.25, 0.25],
                                    E_nuc=[0.4, 0.2, 0.2, 0.2],):
    """
    Creates a casino generator with the given parameters.
    The generator yields a tuple (x,y) where x is the observed sequence
    and y is the hidden state sequence.
    Args:
        sequence_length (int): Length of the sequences to generate.
        batch_size (int): Number of sequences to generate in each batch.
        p_enter_E0 (float): P(E0 | IR)
        p_enter_IR (float): P(IR | E2)
    """
    s, n = 4, 4
    alphabet = list(range(s))
    states = list(range(n))
    A = np.array([[1.-p_enter_E0, p_enter_E0, 0., 0.], 
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                    [p_enter_IR, 1.-p_enter_IR, 0., 0.]])
    B = np.array([IR_nuc, E_nuc, E_nuc, E_nuc])

    def casino_generator():

        x = np.zeros((batch_size, sequence_length, s), dtype=np.float32)
        y = np.zeros((batch_size, sequence_length, n), dtype=np.float32)

        for j in range(batch_size):
            q = 0
            c = np.random.choice(alphabet, p=B[q])
            x[j, 0, c] = 1    
            y[j, 0, q] = 1
            for i in range(1, sequence_length):
                q = np.random.choice(states, p=A[q])
                c = np.random.choice(alphabet, p=B[q])
                x[j, i, c] = 1    
                y[j, i, q] = 1
        yield x,y
    return casino_generator