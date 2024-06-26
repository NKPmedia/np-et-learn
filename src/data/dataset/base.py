import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BasePckHdf5Loader(Dataset):

    def __init__(self, path, preload: bool = False, overwrite_size: int = None):

        self.overwrite_size = int(overwrite_size) if overwrite_size is not None else None
        file_name, file_extension = os.path.splitext(path)
        if file_extension == ".hdf5":
            f = h5py.File(path, 'r')
            self.db = {}
            for name in f:
                if preload:
                    self.db[name] = torch.from_numpy(f[name][:])
                else:
                    self.db[name] = f[name]
        elif file_extension == ".pck":
            dbfile = open(path, 'rb')
            self.db = pickle.load(dbfile)
        else:
            raise Exception("Unknown extension")

        self.preload = preload

        idx_file_path = os.path.join(os.path.dirname(path), "test_idx_context_data.npz")
        if os.path.isfile(idx_file_path):
            self.idx_data = np.load(idx_file_path, allow_pickle=True)  #

    def __len__(self):
        if self.overwrite_size is not None:
            return self.overwrite_size
        return self.db["x"].shape[0]

    def __getitem__(self, item):
        return self.db[item]

    def to(self, device):
        if self.preload:
            for name in self.db.keys():
                self.db[name] = self.db[name].to(device)
        else:
            raise Exception("Cant transfer non preloaded data to a device")
