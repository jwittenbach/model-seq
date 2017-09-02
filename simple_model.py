import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic
from model import BatchModel

class SimpleModel(BatchModel):

    params = ["C", "G", "a", "b"]

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

        X = tf.matmul(C, G)

        # from this point on, everything is element-wise
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

if __name__ == "__main__":
    import numpy as np
    from train import cv_batch_fit

    n_cells, n_genes, k = 10, 5, 3
    model = SimpleModel(n_cells, n_genes, k)

    C = np.random.randn(n_cells, k)
    G = np.random.randn(k, n_genes)
    a = 3
    b = 1
    print(np.exp(np.dot(C, G)))
    data = model.sample(batch_inds=np.arange(n_cells*n_genes, dtype='int64'), C=C, G=G, a=a, b=b)
    data = data.reshape(n_cells, n_genes)
    print(data)
    result = cv_batch_fit(model, data, cv_frac=0.05, batch_frac=0.2)
    print(result)
