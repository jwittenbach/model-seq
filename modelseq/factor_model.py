import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic

from modelseq.model import BatchModel


class FactorModel(BatchModel):

    params = ["C", "G", "s_c", "c_g", "a", "b"]

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
        s_c = tf.Variable(tf.random_normal((self.n_cells, 1)), name="s_c")
        s_g = tf.Variable(tf.random_normal((1, self.n_genes)), name="s_g")

        X = tf.matmul(C, G) + s_c + s_g

        # extract matrix elements for mini-batch
        X_sample = tf.boolean_mask(X, self.batch_mask)
        n_samples = tf.shape(X_sample)[0]

        # simple non-linearity
        M_sample = tf.exp(X_sample)

        # per-element categorical variables with probs [p(X), 1-p(X)]
        # where p(X) is a sigmoid
        a = tf.Variable(1.0, name="a")
        b = tf.Variable(-1.0, name="b")
        X_scaled = b*(X_sample - a)
        p = tf.sigmoid(X_scaled)
        probs = tf.stack([1-p, p], axis=-1)
        cat = Categorical(probs=probs)

        # counts are mixture model between Poisson(M) (signal) and 0 (dropout)
        signal = Poisson(M_sample) 
        dropout = Deterministic(0.0*tf.ones((n_samples,)))
        counts = Mixture(cat, [signal, dropout])
        
        return counts
