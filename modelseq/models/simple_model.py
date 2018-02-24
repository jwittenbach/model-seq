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
        C = tf.Variable(tf.random_uniform((self.shape[0], self.k)) - 0.5, name="C")
        G = tf.Variable(tf.random_uniform((self.k, self.shape[1])) - 0.5, name="G")
        #mu = tf.Variable(tf.zeros((1,))[0], name='mu')
        mu_c = tf.Variable(tf.zeros((self.shape[0], 1)), name='mu_c')
        mu_g = tf.Variable(tf.zeros((1, self.shape[1])), name='mu_g')

        X = mu_g + mu_c + tf.matmul(C, G, name="X")

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

        #counts = Poisson(M_sample)
        
        return counts


    @property
    def loss(self):
        """
        Override
        """
        vars = self.get_variables()
        norm_C = tf.norm(vars['C'])
        norm_G = tf.norm(vars['G'])
        #norm_mu_c = tf.norm(vars['mu_c'])
        norm_mu_g = tf.norm(vars['mu_g'])
        return self.ml_loss + self.alpha * (norm_C + norm_G)
