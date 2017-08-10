import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic
from model import BatchModel

class FactorModel(BatchModel):

    def __init__(self, n_cells, n_genes, k):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.k = k

        self.shape = (n_cells, n_genes)

        super().__init__()

    def _generative_model(self):

        # low-rank factorization
        C = tf.Variable(tf.random_uniform((self.n_cells, self.k)))
        G = tf.Variable(tf.random_uniform((self.k, self.n_genes)))
        self.params['C'] = C
        self.params['G'] = G

        X = tf.matmul(C, G)

        # from this point on, everything is element-wise
        #X_flat = tf.reshape(X, (-1,))
        # batch/CV sample 
        #X_sample = tf.gather(X_flat, self.batch_inds)
        X_sample = tf.gather_nd(X, self.batch_inds)
        
        # cell- or gene-specific factors
        #S = tf.nn.softplus(tf.Variable(tf.random_uniform((n_cells,))))
        #self.params['S'] = S
        #S = tf.reshape(S, (-1, 1))

        # simple non-linearity
        M_sample = tf.exp(X_sample)+1 

        # per-element categorical variables with probs [p, 1-p]
        p = tf.sigmoid(tf.Variable(tf.random_uniform((1,))[0]))
        self.params['p'] = p
        probs = tf.tile([1-p, p], (self.batch_size,))
        probs = tf.reshape(probs, (-1, 2))
        cat = Categorical(probs=probs)

        # counts are mixture model between Poisson(M) (signal) and 0 (dropout)
        signal = Poisson(M_sample) 
        dropout = Deterministic(0.0*tf.ones((self.batch_size,)))
        counts = Mixture(cat, [signal, dropout])
        
        return counts 

if __name__ == "__main__":
    import numpy as np
    from train import cv_batch_fit

    n_cells, n_genes, k = 10, 5, 3
    model = FactorModel(n_cells, n_genes, k)

    C = np.random.randn(n_cells, k)
    G = np.random.randn(k, n_genes)
    print(np.exp(np.dot(C, G)))
    data = model.sample(batch_inds=np.arange(n_cells*n_genes, dtype='int64'), C=C, G=G, p=0.0)
    data = data.reshape(n_cells, n_genes)
    print(data)
    result = cv_batch_fit(model, data, cv_frac=0.05, batch_frac=0.2)
    print(result)
