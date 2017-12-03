import numpy as np
import tensorflow as tf


class MLFactorModel(object):

    def __init__(self, shape, session=None):
        self.shape = shape

        # placeholders minibatch information
        self.batch_mask = tf.placeholder(tf.bool, self.shape)
        self.batch_targets = tf.placeholder(tf.float32, (None,))

        # make the computational graph for the forward model
        self.estimate = self._generative_model()

        self.session = tf.Session() if session is None else session
        self.session.run(tf.global_variables_initializer())

    @property
    def parameters(self):
        return {k: self.session.run(v) for k, v in self.get_variables().items()}

    def sample(self, batch_mask=None, **params):
        feed_dict = self._get_feed_dict(batch_mask=batch_mask, **params)
        result = self.session.run(self.estimate.sample(), feed_dict)
        if batch_mask is None:
            result = result.reshape(self.shape)
        return result

    def _get_feed_dict(self, batch_mask=None, **params):
        variable_map = self.get_variables()
        feed_dict = {}
        for k, v in params.items():
            if k in variable_map.keys():
                feed_dict[variable_map[k]] = v
            else:
                print("Error: '{}' is not a valid parameter name")
                break
        batch_mask = np.ones(self.shape, dtype=bool) if batch_mask is None else batch_mask
        feed_dict[self.batch_mask] = batch_mask
        return feed_dict

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path + "/ckpt")

    @classmethod
    def get_variables(cls):
        return {v.name.split(":")[0]: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}

    @classmethod
    def load_variables(cls, path):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + "/ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint(path))
            values = {k: sess.run(v) for k, v in cls.get_variables().items()}
        return values

    @property
    def ml_loss(self):
       return -tf.reduce_mean(self.estimate.log_prob(self.batch_targets))

    @property
    def loss(self):
        """
        For an implementation of this class, this functional can be overridden
        to modify the loss (e.g. add regularization terms)

        The property should return tensor representing the loss.
        """
        return self.ml_loss

    def _generative_model(self):
        """
        To implement an actual instance of this class, this function must be
        overridden.

        This function should define the generative model by building a
        computational graph, the output of which is a random variable that
        models the data
        - any Variables that represenent parameters to be fit should be added
          to the dictionary self.params['param_name'] = Variable
        - the function should return the Tensor that models the data
        """
        raise NotImplementedError


