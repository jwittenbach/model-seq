#from test_model import FactorModel
from simple_model import SimpleModel
from train import cv_batch_fit
from scipy.io import mmread

print("loading data...")
path = "/home/jason/Documents/matrix.mtx"
data = mmread(path).toarray().T
print("data shape: ", data.shape)

print("building model...")
n_cells, n_genes = data.shape
k = 3
model = SimpleModel(n_cells, n_genes, k)

print("fitting model...")
result = cv_batch_fit(model, data, cv_frac=0.01, batch_frac=0.01)
print("\n", result)
