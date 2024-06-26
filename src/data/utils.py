import torch
from torch.utils.data import default_collate


def _up_rank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')


def custom_collate_fn(batch):
    """Batch is created by the specified Sampler.
    Workaround to ensure that the length in each batch is always identical!"""
    return batch[0]


def different_context_size_collate_fn(batch):
    """
    extends the context size to the largest context size in the batch
    extends with zeros

    generates a list with the real number of context points for each batch element

    also extends the target size to the largest target size in the batch
    extends with zeros

    generates a list with the real number of target points for each batch element
    """
    max_context_size = max([elem["x_context"].size(-2) for elem in batch])
    max_target_size = max([elem["x_target"].size(-2) for elem in batch])

    for elem in batch:
        context_size = elem["x_context"].size(-2)
        target_size = elem["x_target"].size(-2)
        elem["x_context"] = torch.constant_pad_nd(elem["x_context"], (0, 0, 0, max_context_size - context_size))
        elem["y_context"] = torch.constant_pad_nd(elem["y_context"], (0, 0, 0, max_context_size - context_size))
        elem["x_target"] = torch.constant_pad_nd(elem["x_target"], (0, 0, 0, max_target_size - target_size))
        elem["y_target"] = torch.constant_pad_nd(elem["y_target"], (0, 0, 0, max_target_size - target_size))
        elem["context_sizes"] = context_size
        elem["target_sizes"] = target_size

    return default_collate(batch)