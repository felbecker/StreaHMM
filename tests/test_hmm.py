import casino
import mini_gene_pred
import numpy as np


def test_casino():
    print("Testing casino generator")
    generator = casino.make_casino_generator(sequence_length=100, 
                                             batch_size=1, 
                                             p_honest=0.95,
                                             p_dishonest=0.8,
                                             p_six=0.9)
    x, y = next(generator())
    print(np.argmax(x, -1)+1, np.argmax(y, -1))


def test_mini_gene_pred():
    print("Testing mini gene prediction generator")
    generator = mini_gene_pred.make_gene_structure_generator(sequence_length=100, batch_size=1)
    x, y = next(generator())
    print(np.argmax(x, -1), np.argmax(y, -1))