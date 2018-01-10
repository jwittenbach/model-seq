import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic

from modelseq.model import MLFactorModel


class SimpleModel(MLFactorModel):

    def __init__(self, k, alpha, *args, **kwargs):
        self.k = k
        self.alpha = alpha
        super().__init__(*args, **kwargs)


    def _generative_model(self):
        """
        Required override
        """

        # low-rank factorization
        C = tf.Variable(tf.random_uniform((self.shape[0], self.k)), name="C")
        G = tf.Variable(tf.random_uniform((self.k, self.shape[1])), name="G")

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


    # @property
    # def loss(self):
    #     """
    #     Override
    #     """
    #     return self.ml_loss + self.alpha * tf.norm(self.get_variables()['C'])
