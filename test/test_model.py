import unittest
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    Poisson, Categorical, Mixture, Deterministic

from modelseq.model import MLFactorModel

MODEL_VAR_VALUE = 0.5
MODEL_VAR_NAME = 'p'
RANDOM_VAR_NAME = 'v'

class SimpleModel(MLFactorModel):

    def _generative_model(self):
        ones = tf.constant(np.ones(self.shape).astype(np.float32))
        sample = tf.boolean_mask(ones, self.batch_mask)
        n_samples = tf.shape(sample)[0]

        z1 = Poisson(rate=100.0*sample)
        z2 = Deterministic(0.0*sample)

        p = tf.Variable(MODEL_VAR_VALUE, name=MODEL_VAR_NAME)
        v = tf.Variable(tf.random_uniform([1])[0], name=RANDOM_VAR_NAME)

        probs = tf.tile([[p, 1 - p]], (n_samples, 1))
        cat = Categorical(probs=probs)

        z = Mixture(cat, [z1, z2])

        return z

class TestMLFactorModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestMLFactorModel, cls).setUpClass()

        cls.mask = np.array([
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [False, True, True]
        ])
        cls.shape = cls.mask.shape
        cls.model = SimpleModel(cls.shape)

    def test_shape(self):
        self.assertEqual(self.model.shape, self.shape)

    def test_sample(self):
        sample = self.model.sample(p=0.5)
        self.assertEqual(sample.shape, self.shape)

        sample = self.model.sample(p=0.0)
        self.assertTrue(np.array_equal(sample, np.zeros(self.shape)))

        sample = self.model.sample(batch_mask=self.mask, p=0.5)
        self.assertEqual(sample.shape, (self.mask.sum(),))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dir_name:
            self.model.save(dir_name)
            vars = MLFactorModel.load_variables(dir_name)
        self.assertTrue(MODEL_VAR_NAME in vars)
        self.assertEqual(vars[MODEL_VAR_NAME], MODEL_VAR_VALUE)

    def test_parameters(self):
        self.assertTrue(MODEL_VAR_NAME in self.model.parameters)
        self.assertEqual(self.model.parameters[MODEL_VAR_NAME], MODEL_VAR_VALUE)

    def test_set_parameter(self):
        self.model.set_parameter(MODEL_VAR_NAME, MODEL_VAR_VALUE + 1)
        self.assertEqual(self.model.parameters[MODEL_VAR_NAME], MODEL_VAR_VALUE + 1)
        self.model.set_parameter(MODEL_VAR_NAME, MODEL_VAR_VALUE)
        self.assertEqual(self.model.parameters[MODEL_VAR_NAME], MODEL_VAR_VALUE)

    def test_init_seed(self):
        # this is a little awkward to test because of the way we assume a single
        # session / default graph

        g = tf.Graph()
        with g.as_default():
            model_1 = SimpleModel(self.shape, init_seed=0)
            v_1 = model_1.parameters[RANDOM_VAR_NAME]

        g = tf.Graph()
        with g.as_default():
            model_2 = SimpleModel(self.shape, init_seed=0)
            v_2 = model_2.parameters[RANDOM_VAR_NAME]

        g = tf.Graph()
        with g.as_default():
            model_3 = SimpleModel(self.shape, init_seed=1)
            v_3 = model_3.parameters[RANDOM_VAR_NAME]

        self.assertEqual(v_1, v_2)
        self.assertNotEqual(v_1, v_3)

