from scipy.io import loadmat
import os

data_dir = os.path.dirname(os.path.realpath(__file__))

class DataLoader:
    def load_dataset(dataset_name):
        path = os.path.join(data_dir, dataset_name)
        return loadmat(path)