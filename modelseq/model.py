import tensorflow as tf

class Model(object):

    """
    To implement an actual instance of this class, this class variable should
    be replaced with a list of strings of all model parameter names. Make sure
    that variable names are actually set to these values.
    """
    params = []

    def __init__(self):
        self.estimate = self._generative_model()

    def sample(self, **params):
        feed_dict = self._get_feed_dict(params)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            counts = sess.run(self.estimate.sample(), feed_dict)
        return counts

    @property
    def base_loss(self, data):
        return -tf.reduce_mean(self.estimate.log_prob(data))

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

    def _get_feed_dict(self, params):
        feed_dict = {}
        for k, v in params.items():
            if k in self.params.keys():
                feed_dict[self.params[k]] = v
        return feed_dict

    @classmethod
    def restore(cls, path):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + "ckpt.meta")
            saver.restore(sess, tf.train.latest_checkpoint(path))
            values = {p: sess.run(p + ":0") for p in cls.params}
        return values


class BatchModel(Model):

    def __init__(self, shape):
        self.shape = shape

        # placeholders for data fed in on each training step
        self.batch_mask = tf.placeholder(tf.bool, self.shape)
        self.batch_targets = tf.placeholder(tf.float32, (None,))

        super().__init__()

    @property
    def base_loss(self):
       return -tf.reduce_mean(self.estimate.log_prob(self.batch_targets)) 

    def _get_feed_dict(self, params):
        feed_dict = super()._get_feed_dict(params)
        feed_dict[self.batch_inds] = params["batch_inds"]
        return feed_dict

