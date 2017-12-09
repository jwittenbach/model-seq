import unittest

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

CV_FRAC = 0.2
BATCH_FRAC = 0.1

class TestTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestTrainer, cls).setUpClass()

        cls.shape = (10, 10)
        cls.model = ConstantModel(cls.shape)

        cls.data = cls.model.sample()
        cls.opt = tf.train.GradientDescentOptimizer(0.1)

        cls.train = Trainer(
            model=cls.model, data=cls.data, optimizer=cls.opt, logging=False,
            cv_frac=CV_FRAC, batch_frac=BATCH_FRAC
        )

    def test_masks(self):
        # quick calculations for when batches + CV evenly divide number of elements
        n_elements = int(np.prod(self.shape))
        n_cv = int(CV_FRAC * n_elements)
        n_batch = int(BATCH_FRAC * n_elements)
        n_batches = int((1 - CV_FRAC) / BATCH_FRAC)

        # correct number and shape of masks
        self.assertEqual(len(self.train.batch_masks), n_batches)
        self.assertEqual(len(self.train.batch_targets), n_batches)
        for i in range(n_batches):
            self.assertEqual(self.train.batch_masks[i].shape, self.shape)
        self.assertEqual(self.train.cv_mask.shape, self.shape)

        # correct batch + target structure
        for i in range(n_batches):
            self.assertEqual(self.train.batch_targets[i].shape, (n_batch,))
            self.assertEqual(self.train.batch_masks[i].sum(), n_batch)
        self.assertEqual(self.train.cv_targets.shape, (n_cv,))
        self.assertEqual(self.train.cv_mask.sum(), n_cv)

        all_masks = self.train.batch_masks + [self.train.cv_mask]
        total = np.stack(all_masks).sum(axis=0)
        self.assertTrue(np.array_equal(total, np.ones(self.shape)))

    def test_training(self):

        self.model.set_parameter(MU_NAME, 0.0)

        for _ in range(1000):
            self.train.step()

        a = self.model.parameters[MU_NAME]
        b = self.data.mean()
        d = np.abs((a - b)/b)
        tol = 0.05
        self.assertLess(d, tol)

    def test_seeds(self):

        params = {
            "model": self.model,
            "data": self.data,
            "optimizer": self.opt,
            "cv_frac": CV_FRAC,
            "batch_frac": BATCH_FRAC,
            "batch_seed": 0,
            "logging": False
        }
        train_1 = Trainer(**params)
        train_2 = Trainer(**params)

        self.assertTrue(np.array_equal(train_1.cv_mask, train_2.cv_mask))
        for mask_1, mask_2 in zip(train_1.batch_masks, train_2.batch_masks):
            self.assertTrue(np.array_equal(mask_1, mask_2))

        self.model.set_parameter(MU_NAME, 0.0)

        params["batch_seed"] = 1
        train_3 = Trainer(**params)

        self.assertFalse(np.array_equal(train_1.cv_mask, train_3.cv_mask))
        for mask_1, mask_3 in zip(train_1.batch_masks, train_3.batch_masks):
            self.assertFalse(np.array_equal(mask_1, mask_3))
