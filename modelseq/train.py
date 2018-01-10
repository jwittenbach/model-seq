import logging
import tempfile
from time import strftime

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, model, data, optimizer, cv_frac, batch_frac,
                 logdir=None, logging=True, batch_seed=0):

        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.logging = logging

        self.logdir = "logs/" + strftime("%Y.%m.%d-%H.%M.%S") if logdir is None else logdir

        # ugh
        self.fit = self._get_fit_op()

        # make boolean masks to for CV and batch sets
        masks = self._make_masks(cv_frac, batch_frac, batch_seed)

        cv_mask = masks[0]
        self.cv_feed_dict = {
            self.model.batch_mask: cv_mask,
            self.model.batch_targets: data[cv_mask]
        }

        self.batch_masks = masks[1]
        self.batch_targets = [data[mask] for mask in self.batch_masks]
        self.batch_feed_dict = None
        self.n_batches = len(self.batch_masks)

        # set up monitoring
        if self.logging:
            self.summary_writer, self.summaries = self._prepare_monitoring()

        # initialize slot variables from optimizer

    def _get_fit_op(self):
        saver = tf.train.Saver()
        with tempfile.TemporaryDirectory() as dir_name:
            save_path = saver.save(self.model.session, dir_name+"vars.ckpt")
            fit = self.optimizer.minimize(self.model.loss)
            self.model.session.run(tf.global_variables_initializer())
            saver.restore(self.model.session, save_path)
        return fit



    @property
    def cv_mask(self):
        return self.cv_feed_dict[self.model.batch_mask]

    @property
    def cv_targets(self):
        return self.cv_feed_dict[self.model.batch_targets]

    def _make_masks(self, cv_frac, batch_frac, batch_seed):

        np.random.seed(batch_seed)

        n_elems = np.prod(self.model.shape)
        n_cv = np.floor(cv_frac * n_elems).astype(int)
        n_batch = np.floor(batch_frac * n_elems).astype(int)
        n_batches = np.round((n_elems - n_cv) / n_batch).astype(int)

        inds = np.random.permutation(np.arange(n_elems))

        cv_inds = inds[:n_cv]
        cv_mask = np.zeros(self.model.shape, dtype=bool)
        cv_mask[np.unravel_index(cv_inds, self.model.shape)] = True

        train_inds = inds[n_cv:]
        batches = np.array_split(train_inds, n_batches)
        batch_masks = []
        for i in range(n_batches):
            batch_masks.append(np.zeros(self.model.shape, dtype=bool))
            batch_masks[i][np.unravel_index(batches[i], self.model.shape)] = True

        return cv_mask, batch_masks

    def _prepare_monitoring(self):

        train_loss = tf.summary.scalar("training_loss", self.model.loss)
        test_loss = tf.summary.scalar("testing_loss", self.model.loss)

        variables = []
        for name, variable in self.model.get_variables().items():
            # scalars
            if variable.shape.ndims < 2:
                variables.append(tf.summary.scalar(name, variable))
            # matrices
            else:
                variables.append(tf.summary.histogram(name, variable))

        summaries = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'variables': variables
        }

        summary_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())

        return summary_writer, summaries

    def summarize(self, step, train_loss=True, test_loss=True, variables=False):

        if not self.logging:
            print("warning: trying to summarize when logging has been inactivated")
            return

        if train_loss:
            _, summary = self.model.session.run(
                [self.model.loss, self.summaries["train_loss"]], self.batch_feed_dict
            )
            self.summary_writer.add_summary(summary, step)

        if test_loss:
            _, summary = self.model.session.run(
                [self.model.loss, self.summaries["test_loss"]], self.cv_feed_dict
            )
            self.summary_writer.add_summary(summary, step)

        if variables:
            summaries = self.model.session.run(self.summaries["variables"])
            for summary in summaries:
                self.summary_writer.add_summary(summary, step)

    def step(self):

        batch = np.random.randint(self.n_batches)
        self.batch_feed_dict = {
            self.model.batch_mask: self.batch_masks[batch],
            self.model.batch_targets: self.batch_targets[batch]
        }
        self.model.session.run(self.fit, self.batch_feed_dict)
