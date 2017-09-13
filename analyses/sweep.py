import logging
import os

import numpy as np
from scipy.io import mmread
import tensorflow as tf

from modelseq.train import cv_batch_fit, make_masks
from modelseq.factor_model import FactorModel

# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


logger.info("loading data...")
path = "/home/jason/Documents/matrix.mtx"
data = mmread(path).toarray().T

logger.info("data shape: {}".format(data.shape))
n_cells, n_genes = data.shape

cv_frac = 0.01
batch_frac = 0.01
epochs = 300

n_steps = int(epochs / batch_frac)
cv_mask, batch_masks = make_masks(data.shape, cv_frac=cv_frac, batch_frac=batch_frac)

k_max = 20
results = []

dir = "simple_sweep"
os.mkdir("checkpoints/{}".format(dir))
os.mkdir("logs/{}".format(dir))

for k in np.arange(k_max) + 1:
    logger.info("fitting model with k = {}".format(k))

    model = FactorModel(n_cells, n_genes, k)

    test_loss = cv_batch_fit(
        model=model, data=data, cv_mask=cv_mask, batch_masks=batch_masks,
        n_steps=n_steps, dir="simple_sweep/k{}".format(k)
    )
    results.append(test_loss)
    np.save("checkpoints/{}/results.npy".format(dir), results)
    tf.reset_default_graph()