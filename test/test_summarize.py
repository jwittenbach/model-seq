import unittest

import tensorflow as tf
from tensorflow.contrib.distributions import \
    Normal, Deterministic, Categorical, Mixture
import numpy as np

from modelseq.train import Trainer
from modelseq.model import MLFactorModel


N_SAMPLES = 200
N_COLS = 10
MODEL_VECTOR_VALUE = 5 + np.arange(N_COLS)[np.newaxis, :].astype(np.float32)
MODEL_VECTOR_NAME = "v"
MODEL_PROB_VALUE = 0.2
MODEL_PROB_INV_SIGMOID = -1.3862943611198906
MODEL_PROB_NAME = "p"


class VectorModel(MLFactorModel):

    def _generative_model(self):
        v = tf.Variable(MODEL_VECTOR_VALUE, name=MODEL_VECTOR_NAME)
        p = tf.sigmoid(tf.Variable(MODEL_PROB_INV_SIGMOID, name=MODEL_PROB_NAME))
        sigma = tf.constant(1.0)

        mat = tf.tile(v, (N_SAMPLES, 1))
        mat_batch = tf.boolean_mask(mat, self.batch_mask)
        n_batch = tf.shape(mat_batch)[0]

        probs = tf.tile([1 - p, p], (n_batch,))
        probs = tf.reshape(probs, (-1, 2))
        cat = Categorical(probs=probs)

        n = Normal(loc=mat_batch, scale=sigma)
        d = Deterministic(0.0 * tf.ones((n_batch,)))

        return Mixture(cat, [n, d])


class TestSummarize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestSummarize, cls).setUpClass()

        cls.model = VectorModel((N_SAMPLES, N_COLS))
        cls.data = cls.model.sample()
        cls.opt = tf.train.GradientDescentOptimizer(0.1)

        cls.train = Trainer(
            model=cls.model, data=cls.data, optimizer=cls.opt, cv_frac=0.2, batch_frac=0.1
        )

    def test_summarize(self):

        self.model.set_parameter(MODEL_VECTOR_NAME, np.zeros([1, N_COLS]))
        self.model.set_parameter(MODEL_PROB_NAME, 0.0) # probability = 0.5 (after passing through sigmoid)

        for i in range(20):
            print(i)
            self.train.step()
            self.train.summarize(i, variables=True)

