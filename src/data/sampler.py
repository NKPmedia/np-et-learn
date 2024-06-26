import numpy as np
from torch.utils.data import Sampler


class BatchedSequentialSampler(Sampler):

    def __init__(self, data_source, batch_size):
        super(BatchedSequentialSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        lst = list(range(len(self.data_source)))
        np.random.shuffle(lst)
        return iter(lst)

    def __len__(self) -> int:
        return len(self.data_source)