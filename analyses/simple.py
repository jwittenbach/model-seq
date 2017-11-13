import logging

import numpy as np
from scipy.io import mmread
from modelseq.simple_model import SimpleModel
from modelseq.nb_model import NBModel

from modelseq.train import cv_batch_fit, make_masks


logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


logger.info("loading data...")

# path = "/home/jason/Documents/matrix.mtx"
# data = mmread(path).toarray().T

path = "/home/jason/Documents/sample.npy"
data = np.load(path)

logger.info("data shape: {}".format(data.shape))

logger.info("building model...")
n_cells, n_genes = data.shape
k = 5

cv_frac = 0.01
batch_frac = 0.01
epochs = 300
n_steps = int(epochs / batch_frac)

model = SimpleModel(n_cells, n_genes, k)
#model = NBModel(n_cells, n_genes, k)

logger.info("fitting model...")
cv_mask, batch_masks = make_masks(data.shape, cv_frac=cv_frac, batch_frac=batch_frac)

result = cv_batch_fit(
    model=model, data=data, cv_mask=cv_mask, batch_masks=batch_masks,
    n_steps=n_steps, dir="simple-small-{}".format(k)
)
logger.info("{}".format(result))
