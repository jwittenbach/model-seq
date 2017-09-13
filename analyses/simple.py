import logging

from scipy.io import mmread
from modelseq.simple_model import SimpleModel

from modelseq.train import cv_batch_fit, make_masks


logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


logger.info("loading data...")
path = "/home/jason/Documents/matrix.mtx"
data = mmread(path).toarray().T
logger.info("data shape: {}".format(data.shape))

logger.info("building model...")
n_cells, n_genes = data.shape
k = 3
model = SimpleModel(n_cells, n_genes, k)

logger.info("fitting model...")
cv_mask, batch_masks = make_masks(model.shape, cv_frac=0.01, batch_frac=0.01)

result = cv_batch_fit(
    model, data, cv_mask, batch_masks, n_steps=10000)
logger.info("{}".format(result))
