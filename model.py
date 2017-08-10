import tensorflow as tf

class Model(object):

    def __init__(self):
        self.params = {}
        self.estimate = self._generative_model()

    def sample(self, **params):
        feed_dict = self._get_feed_dict(params)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            counts = sess.run(self.estimate.sample(), feed_dict)
        return counts

    @property
    def loss(self, data):
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

class BatchModel(Model):

    def __init__(self): 
        # placeholders for data fed in on each training step
        self.batch_inds = tf.placeholder(tf.int64, (None, 2))
        self.batch_targets = tf.placeholder(tf.float32, (None,))
        self.batch_size = tf.shape(self.batch_inds)[0]

        super().__init__()

    @property
    def loss(self):
       return -tf.reduce_mean(self.estimate.log_prob(self.batch_targets)) 

    def _get_feed_dict(self, params):
        feed_dict = super()._get_feed_dict(params)
        feed_dict[self.batch_inds] = params["batch_inds"]
        return feed_dict

