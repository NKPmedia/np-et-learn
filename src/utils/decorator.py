import numpy as np
import torch


def argument_torch_to_np(f):
    """
    Decorator to convert the parameters that are torch.Tensor to numpy.ndarray
    Parameters
    ----------
    f

    Returns
    -------

    """
    def wrapper(*args, **kw):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.detach().cpu().numpy())
            else:
                new_args.append(arg)
        new_km = {}
        for key, arg in kw.items():
            if isinstance(arg, torch.Tensor):
                new_km[key] = arg.detach().cpu().numpy()
            else:
                new_km[key] = arg
        return f(*new_args, **new_km)

    return wrapper

def argument_remove_channel_dim(f):
    """
    Decorator to remove the last dim (channel dim) from every tensor and ndarray
    Parameters
    ----------
    f

    Returns
    -------

    """
    def wrapper(*args, **kw):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray):
                new_args.append(arg.squeeze(-1))
            else:
                new_args.append(arg)
        new_km = {}
        for key, arg in kw.items():
            if isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray):
                new_km[key] = arg.squeeze(-1)
            else:
                new_km[key] = arg
        return f(*new_args, **new_km)

    return wrapper