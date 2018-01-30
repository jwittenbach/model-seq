import logging

import numpy as np
import tensorflow as tf

from modelseq.models.simple_model import SimpleModel
from modelseq.train import Trainer

logger = logging.getLogger(__name__)
log_format = "%(levelname)s:%(name)s:%(message)s"
# logging.basicConfig(format=log_format, level=logging.INFO)
logging.basicConfig(format=log_format, level=logging.DEBUG)

log_path = "logs/simple/"

# data
logger.info("loading data...")

# data_path = "/Users/jason/Documents/matrix.mtx"
# data = mmread(path).toarray().T

data_path = "/Users/jason/Documents/sampled.npy"
data = np.load(data_path)

logger.info("data shape: {}".format(data.shape))

# model
logger.info("specifying model...")

model = SimpleModel(shape=data.shape, k=5, alpha=0.01)

# mean_counts_per_cell = data.sum(axis=1).mean()
# mu_init = np.tile(mean_counts_per_cell, data.shape[0])[:, np.newaxis]
# model.set_parameter('mu', mu_init)

# training details
logger.info("setting up training...")

opt = tf.train.AdamOptimizer()

train = Trainer(
    model=model, data=data, optimizer=opt, cv_frac=0.1, batch_frac=0.9, logdir=log_path
)

epochs = 100000
n_steps = epochs * train.n_batches
logger.info("epochs:\t{}".format(epochs))
logger.info("steps:\t{}".format(n_steps))

# training loop
logger.info("fitting model...")

vars = model.parameters
# logger.info("mu:\t{}".format(vars['mu']))
logger.debug("C max:\t{}".format(vars['C'].max()))
logger.debug("C min:\t{}".format(vars['C'].min()))
logger.debug("G max:\t{}".format(vars['G'].max()))
logger.debug("G min:\t{}".format(vars['G'].min()))
X = np.dot(vars['C'], vars['G'])
logger.debug("X max:\t{}".format(X.max()))
logger.debug("X min:\t{}".format(X.min()))
for i in range(n_steps):
    if i % 10 == 0:
        print("step:\t{}".format(i))
        train.step()
        train.summarize(i, variables=True)

model.save(log_path)
logger.debug(model.parameters)
