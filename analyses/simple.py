import logging

import numpy as np
from scipy.io import mmread
import tensorflow as tf

from modelseq.models.simple_model import SimpleModel
from modelseq.train import Trainer

logger = logging.getLogger(__name__)
log_format = "%(levelname)s:%(name)s:%(message)s"
# logging.basicConfig(format=log_format, level=logging.INFO)
logging.basicConfig(format=log_format, level=logging.DEBUG)

log_path = "logs/simple-10/"

# data
logger.info("loading data...")

#data_path = "/Users/jason/Documents/matrix.mtx"
data_path = "/home/jason/Documents/matrix.mtx"
data = mmread(data_path).toarray().T

#data_path = "/Users/jason/Documents/sampled.npy"
#data_path = "/home/jason/Documents/sampled.npy"
#data = np.load(data_path)

logger.info("data shape: {}".format(data.shape))

# model
logger.info("specifying model...")

model = SimpleModel(shape=data.shape, k=10, alpha=0.00)

# mean_counts = data.mean()
# model.set_parameter('mu', mean_counts)

#mean_counts_per_gene = data.mean(axis=0)[np.newaxis, :]
#model.set_parameter('mu_g', np.log(mean_counts_per_gene))

# training details
logger.info("setting up training...")

opt = tf.train.AdamOptimizer(0.1)

train = Trainer(
    model=model, data=data, optimizer=opt, cv_frac=0.1, batch_frac=0.9, logdir=log_path
)

epochs = 10000000
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
#logger.debug("mu_c max:\t{}".format(vars["mu_c"].max()))
#logger.debug("mu_c min:\t{}".format(vars["mu_c"].min()))
#logger.debug("mu_g max:\t{}".format(vars["mu_g"].max()))
#logger.debug("mu_c min:\t{}".format(vars["mu_c"].min()))
#logger.debug("mu:\t{}".format(vars["mu"]))
#X = vars["mu"] + vars["mu_c"] + vars["mu_g"] + np.dot(vars['C'], vars['G'])
# logger.debug("X max:\t{}".format(X.max()))
# logger.debug("X min:\t{}".format(X.min()))
for i in range(n_steps):
    if i % 100000 == 0:
        print("step:\t{}".format(i))
        train.step()
#        X_tensor = tf.get_default_graph().get_tensor_by_name('X:0')
#        X = model.session.run(X_tensor)
#        print("X max:\t{}".format(X.max()))
#        print("X min:\t{}".format(X.min()))
#        print("mu:\t{}".format(model.parameters["mu"]))
        train.summarize(i, variables=True)

model.save(log_path)
#logger.debug(model.parameters)
