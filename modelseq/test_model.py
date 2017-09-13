import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    NegativeBinomial, Categorical, Mixture, Deterministic

from modelseq.model import Model


class TestModel(Model):

    def __init__(self, n_cells, n_genes):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.shape = (n_cells, n_genes)
        super().__init__()

    def _generative_model(self):
    
        ones = np.ones(self.shape).astype(np.float32)
        z1 = NegativeBinomial(100.0, probs=0.5*ones)
        z2 = Deterministic(0.0*ones)
        
        p = tf.Variable(0.5)
        self.params['p'] = p

        probs = tf.tile([[p, 1-p]], self.shape)
        probs = tf.reshape(probs, self.shape + (2,))
        cat = Categorical(probs=probs)
    
        z = Mixture(cat, [z1, z2])
    
        return z
