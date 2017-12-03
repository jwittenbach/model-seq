import unittest
import tempfile

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np

from modelseq.train import Trainer
from modelseq.model import MLFactorModel

MU_VAL = 10.0
MU_NAME = 'mu'

class ConstantModel(MLFactorModel):

    def _generative_model(self):
        mu = tf.Variable(MU_VAL, name=MU_NAME)
        ones = tf.constant(np.ones(self.shape).astype(np.float32))
        mat = mu * ones

        sigma = tf.constant(1.0)
        return Normal(loc=mat, scale=sigma)

class TestTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestTrainer, cls).setUpClass()

        cls.shape = (10, 10)
        cls.model = ConstantModel(cls.shape)

    def test_shape(self):
        self.assertEqual(self.model.shape, self.shape)
