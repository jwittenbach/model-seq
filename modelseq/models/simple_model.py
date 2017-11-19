import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic

from modelseq.model import BatchModel


class SimpleModel(BatchModel):

    params = ["C", "G", "p"]

    def __init__(self, n_cells, n_genes, k):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.k = k

        shape = (n_cells, n_genes)
        super().__init__(shape)

    def _generative_model(self):

        # low-rank factorization
        C = tf.Variable(tf.random_uniform((self.n_cells, self.k)), name="C")
        G = tf.Variable(tf.random_uniform((self.k, self.n_genes)), name="G")

        X = tf.matmul(C, G, name="X")

        # extract matrix elements for mini-batch
        X_sample = tf.boolean_mask(X, self.batch_mask)
        n_samples = tf.shape(X_sample)[0]

        # simple non-linearity
        M_sample = tf.exp(X_sample)

        # per-element categorical variables with probs [p, 1-p]
        p = tf.sigmoid(tf.Variable(tf.random_uniform((1,))[0]), name="p")
        probs = tf.tile([1 - p, p], (n_samples,))
        probs = tf.reshape(probs, (-1, 2))
        cat = Categorical(probs=probs)

        # counts are mixture model between Poisson(M) (signal) and 0 (dropout)
        signal = Poisson(M_sample) 
        dropout = Deterministic(0.0*tf.ones((n_samples,)))
        counts = Mixture(cat, [signal, dropout])
        
        return counts
