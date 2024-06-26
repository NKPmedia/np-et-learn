import json
import os
import pickle
from os import path
from typing import List, Dict

import h5py
import jsonpickle
import numpy as np

from src.data.dataset.dynamic_sys_dataset import DynamicSystemDatasetGenerator, OfflineDynamicSystemDataset
from src.data.utils import custom_collate_fn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from src.utils.torch_utils import t2np
import copy


class DynamicSystemDataModule(LightningDataModule):
    def __init__(self,
                 data_generation_kwargs: Dict,
                 offline_dataset_kwargs: Dict,
                 batch_size: int = 32,
                 total_dataset_sizes: List[int] = [5000, 200, 200],
                 pregenerated_data_path: str = None,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.total_dataset_sizes = total_dataset_sizes
        self.pregenerated_data_path = pregenerated_data_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_generation_kwargs = data_generation_kwargs
        self.offline_dataset_kwargs = offline_dataset_kwargs
        self.batch_size = batch_size

        self.test_sets = []

        self.format = "hdf5"

    def generate_datasets(self):
        os.makedirs(self.pregenerated_data_path, exist_ok=True)

        for name, size in zip(["train", "val", "test"], self.total_dataset_sizes):
            dset = DynamicSystemDatasetGenerator(
                number_samples=size,
                progress_bar=True,
                **self.data_generation_kwargs)
            data = dset[range(size)]
            data = {key: value.numpy() for key, value in data.items()}
            if self.format == "pickel":
                if path.isfile(path.join(self.pregenerated_data_path, f"{name}.pck")):
                    os.remove(path.join(self.pregenerated_data_path, f"{name}.pck"))
                dbfile = open(path.join(self.pregenerated_data_path, f"{name}.pck"), 'x')
                pickle.dump(data, dbfile)
                dbfile.close()
            else:
                f = h5py.File(path.join(self.pregenerated_data_path, f"{name}.hdf5"), 'w-')
                for key, value in data.items():
                    hdf5_dset = f.create_dataset(key, value.shape, dtype=np.float32, fletcher32=True)
                    hdf5_dset[...] = t2np(value)
                f.close()
            param_path = path.join(self.pregenerated_data_path, f"{name}.json")
            frozen = jsonpickle.encode(dset.generation_arguments_snapshot, indent=4, separators=(',', ': '))
            with open(param_path, 'w') as f:
                f.write(frozen)

    def setup(self, stage: str):
        if self.format == "pickel":
            ext = "pck"
        else:
            ext = "hdf5"
        dataset_path = self.pregenerated_data_path
        if not os.path.isfile(os.path.join(dataset_path, f"train.{ext}")):
            self.generate_datasets()
        if os.path.isfile(os.path.join(dataset_path, f"train.{ext}")):
            self.training_set = OfflineDynamicSystemDataset(
                path=path.join(dataset_path, f"train.{ext}"),
                val_set=False,
                **self.offline_dataset_kwargs)

            self.validation_set = OfflineDynamicSystemDataset(
                path=path.join(dataset_path, f"val.{ext}"),
                val_set=True,
                **self.offline_dataset_kwargs)
        else:
            raise Exception(f"Datasetpath {dataset_path} is empty")

    def train_dataloader(self):
        PARAMS = {'sampler': BatchSampler(RandomSampler(range(len(self.training_set))),
                                          self.batch_size, drop_last=True),
                  'collate_fn': custom_collate_fn}
        return DataLoader(self.training_set, num_workers=self.num_workers, pin_memory=self.pin_memory, **PARAMS)

    def val_dataloader(self):
        PARAMS = {'sampler': BatchSampler(SequentialSampler(range(len(self.validation_set))),
                                          self.batch_size, drop_last=True),
                  'collate_fn': custom_collate_fn}
        return DataLoader(self.validation_set, num_workers=self.num_workers, pin_memory=self.pin_memory, **PARAMS)
