from typing import Tuple, List, Sequence

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module, Parameter, ParameterList


class BaseNormalizerAndEncoder(Module):
    """
    Base class for normalizer and encoder
    The goal of this is to provide normalized inputs to the neural network
    that are encoded such that they  can be used by the nn

    Should be used in the network model so that one can use the normal state of a dynamic system as the input
    """

    def __init__(self):
        super().__init__()
        self.extra_dims = 0

    def encoding_and_normalization_XcYcXt(self, context_x, context_y, target_x):
        """
        Encodes and normalizes the input of the neural network
        Zero mean and unit variance
        Args:
            *args:

        Returns:

        """
        return context_x, context_y, target_x

    def encoding_and_normalization_Xt(self, target_x):
        raise NotImplementedError()

    def encoding_and_normalization_Yt(self, target_y):
        raise NotImplementedError()

    def normalize_context_increment(self, increment):
        raise NotImplementedError()

    def normalize_backwards_increment(self, increment):
        raise NotImplementedError()

    def encoding_and_normalization_XcYcXtYt(self, context_x, context_y, target_x, target_y):
        """
        Encodes and normalizes the input of the neural network
        Zero mean and unit variance
        Args:
            *args:

        Returns:

        """
        return context_x, context_y, target_x, target_y

    def encoding_and_normalization_XcYc(self, context_x, context_y):
        """
        Encodes and normalizes the input of the neural network
        Zero mean and unit variance
        Args:
            *args:

        Returns:

        """
        return context_x, context_y

    def normalize_increment(self, increment):
        """
        Normalizes the increment of the neural network to zero mean and unit variance
        Used when a neural network predicts the change of the state of a dynamic system
        Args:
            *args:

        Returns:

        """
        return increment

    def normalize_output(self, output):
        """
        Normalizes the output of the neural network (reverts zero mean and unit variance)
        Used when a neural network directly predicts the state of a dynamic system
        Args:
            *args:

        Returns:

        """
        return output


class InvPNormalizerAndEncoder(BaseNormalizerAndEncoder):

    def __init__(self, inp_std, inp_mean, inc_std, inc_mean):
        super().__init__()
        self.inp_std = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inp_std])
        self.inp_mean = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inp_mean])
        self.inc_std = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inc_std])
        self.inc_mean = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inc_mean])
        self.extra_dims = 1

    def encoding_and_normalization_XcYcXt(self, context_x, context_y, target_x):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)
        target_x_enc = self.sin_cos_angle_encoding(target_x)
        context_x_enc = (context_x_enc - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y_enc - self.inp_mean[1]) / self.inp_std[1]
        target_x_enc = (target_x_enc - self.inp_mean[2]) / self.inp_std[2]
        return context_x_enc, context_y_enc, target_x_enc

    def encoding_and_normalization_XcYcXtYt(self, context_x, context_y, target_x, target_y):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)
        target_x_enc = self.sin_cos_angle_encoding(target_x)
        target_y_enc = self.sin_cos_angle_encoding(target_y)
        context_x_enc = (context_x_enc - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y_enc - self.inp_mean[1]) / self.inp_std[1]
        target_x_enc = (target_x_enc - self.inp_mean[2]) / self.inp_std[2]
        target_y_enc = (target_y_enc - self.inp_mean[3]) / self.inp_std[3]
        return context_x_enc, context_y_enc, target_x_enc, target_y_enc

    def encoding_and_normalization_XcYc(self, context_x, context_y):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)
        context_x_enc = (context_x_enc - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y_enc - self.inp_mean[1]) / self.inp_std[1]
        return context_x_enc, context_y_enc

    def normalize_increment(self, increment):
        increment = (increment - self.inc_mean[0]) / self.inc_std[0]
        return increment

    @classmethod
    def sin_cos_angle_encoding(cls, state: Float[torch.Tensor, "batch context state"]) -> \
            Float[torch.Tensor, "batch context state+1"]:
        """
        Encoded the angle (in rad) in the state using sin and cos (two channels)
        Expects the angle to be in the third dimension
        
        Args:
            state: State to be encoded; shape(batch_size, context_size, state_size)
        
        https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
        Used in https://academic.oup.com/bioinformatics/article/33/18/2842/3738544?login=false#118817095
        """
        sin_cos_encoding = torch.cat([torch.sin(state[:, :, 2, None]), torch.cos(state[:, :, 2, None])], dim=-1)
        encoded_data = torch.cat([state[:, :, :2], sin_cos_encoding, state[:, :, 3:]], dim=2)

        return encoded_data


def update_and_normalize(data: Float[Tensor, "batch context features"],
                         mean: ParameterList,
                         std: ParameterList,
                         idx: int,
                         alpha: float,
                         train: bool = False,
                         adapt: bool = False):
    """
    Updates the mean and std of the data
    And normalizes the data
    Args:
        data: data to be updated
        mean: mean of the data
        std: std of the data
        idx: index of the data to be updated
        alpha: learning rate

    Returns:

    """
    if train and adapt:
        mean[idx] = (1 - alpha) * mean[idx] + alpha * data.mean(dim=(0, 1))
        data - mean[idx]
        std[idx] = (1 - alpha) * std[idx] + alpha * data.std(dim=(0, 1))
        return data / std[idx]

    return (data - mean[idx]) / std[idx]


def update_and_denormalize(data: Float[Tensor, "batch context features"],
                           mean: ParameterList,
                           std: ParameterList,
                           idx: int,
                           alpha: float,
                           train: bool = False,
                           adapt: bool = False):
    """
    Updates the mean and std of the data
    And normalizes the data
    Args:
        data: data to be updated
        mean: mean of the data
        std: std of the data
        idx: index of the data to be updated
        alpha: learning rate

    Returns:

    """
    if train and adapt:
        data = data * std[idx]
        std[idx] = (1 - alpha) * std[idx] + alpha * data.std(dim=(0, 1))
        data = data + mean[idx]
        mean[idx] = (1 - alpha) * mean[idx] + alpha * data.mean(dim=(0, 1))
        return data

    return (data * std[idx]) + mean[idx]


class HalfCheethaNormalizerAndEncoder(BaseNormalizerAndEncoder):

    def __init__(self, inp_size, inc_size, alpha=0.001, online_adaptation: bool = False):
        super().__init__()
        self.inp_std = ParameterList([Parameter(torch.ones(tmp), requires_grad=False) for tmp in inp_size])
        self.inp_mean = ParameterList([Parameter(torch.zeros(tmp), requires_grad=False) for tmp in inp_size])
        self.inc_std = ParameterList([Parameter(torch.ones(tmp), requires_grad=False) for tmp in inc_size])
        self.inc_mean = ParameterList([Parameter(torch.zeros(tmp), requires_grad=False) for tmp in inc_size])
        self.back_inc_std = ParameterList([Parameter(torch.ones(tmp), requires_grad=False) for tmp in inc_size])
        self.back_inc_mean = ParameterList([Parameter(torch.zeros(tmp), requires_grad=False) for tmp in inc_size])
        self.extra_dims = 0  # 1 angle - 1 x pos

        self.alpha = alpha
        self.online_adaptation = online_adaptation

    def precompute_normalization_values(self, states, controls):
        # context x y target x y
        encoded_states = self.sin_cos_angle_encoding(states)
        increments = states[:, 1:, :] - states[:, :-1, :]
        back_increments = states[:, :-1, :] - states[:, 1:, :]
        state_mean = encoded_states.mean(dim=(0, 1))
        state_std = encoded_states.std(dim=(0, 1))
        action_mean = controls.mean(dim=(0, 1))
        action_std = controls.std(dim=(0, 1))
        increment_mean = increments.mean(dim=(0, 1))
        increment_std = increments.std(dim=(0, 1))
        back_increment_mean = back_increments.mean(dim=(0, 1))
        back_increment_std = back_increments.std(dim=(0, 1))

        self.inp_mean[0] = Parameter(torch.concatenate([state_mean, action_mean], dim=-1), requires_grad=False)
        self.inp_std[0] = Parameter(torch.concatenate([state_std, action_std], dim=-1), requires_grad=False)
        self.inp_mean[1] = Parameter(state_mean, requires_grad=False)
        self.inp_std[1] = Parameter(state_std, requires_grad=False)
        self.inp_mean[2] = Parameter(torch.concatenate([state_mean, action_mean], dim=-1), requires_grad=False)
        self.inp_std[2] = Parameter(torch.concatenate([state_std, action_std], dim=-1), requires_grad=False)
        self.inp_mean[3] = Parameter(state_mean, requires_grad=False)
        self.inp_std[3] = Parameter(state_std, requires_grad=False)
        self.inc_mean[0] = Parameter(increment_mean, requires_grad=False)
        self.inc_std[0] = Parameter(increment_std, requires_grad=False)
        self.back_inc_mean[0] = Parameter(back_increment_mean, requires_grad=False)
        self.back_inc_std[0] = Parameter(back_increment_std, requires_grad=False)

    def get_norm_values(self):
        val_dict = {
            "inp_mean": self.inp_mean,
            "inp_std": self.inp_std,
            "inc_mean": self.inc_mean,
            "inc_std": self.inc_std,
            "back_inc_mean": self.back_inc_mean,
            "back_inc_std": self.back_inc_std,
        }
        return val_dict

    def get_normalized_target_increment(self, inc):
        return (inc - self.inc_mean[0]) / self.inc_std[0]

    def encoding_and_normalization_XcYcXt(self, context_x, context_y, target_x):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)
        target_x_enc = self.sin_cos_angle_encoding(target_x)

        context_x_enc = update_and_normalize(context_x_enc, self.inp_mean, self.inp_std, 0, self.alpha, self.training,
                                             self.online_adaptation)
        context_y_enc = update_and_normalize(context_y_enc, self.inp_mean, self.inp_std, 1, self.alpha, self.training,
                                             self.online_adaptation)
        target_x_enc = update_and_normalize(target_x_enc, self.inp_mean, self.inp_std, 2, self.alpha, self.training,
                                            self.online_adaptation)
        return context_x_enc, context_y_enc, target_x_enc

    def encoding_and_normalization_XcYcXtYt(self, context_x, context_y, target_x, target_y):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)
        target_x_enc = self.sin_cos_angle_encoding(target_x)
        target_y_enc = self.sin_cos_angle_encoding(target_y)

        context_x_enc = update_and_normalize(context_x_enc, self.inp_mean, self.inp_std, 0, self.alpha, self.training,
                                             self.online_adaptation)
        context_y_enc = update_and_normalize(context_y_enc, self.inp_mean, self.inp_std, 1, self.alpha, self.training,
                                             self.online_adaptation)
        target_x_enc = update_and_normalize(target_x_enc, self.inp_mean, self.inp_std, 2, self.alpha, self.training,
                                            self.online_adaptation)
        target_y_enc = update_and_normalize(target_y_enc, self.inp_mean, self.inp_std, 3, self.alpha, self.training,
                                            self.online_adaptation)

        return context_x_enc, context_y_enc, target_x_enc, target_y_enc

    def encoding_and_normalization_XcYc(self, context_x, context_y):
        context_x_enc = self.sin_cos_angle_encoding(context_x)
        context_y_enc = self.sin_cos_angle_encoding(context_y)

        context_x_enc = update_and_normalize(context_x_enc, self.inp_mean, self.inp_std, 0, self.alpha, self.training,
                                             self.online_adaptation)
        context_y_enc = update_and_normalize(context_y_enc, self.inp_mean, self.inp_std, 1, self.alpha, self.training,
                                             self.online_adaptation)
        return context_x_enc, context_y_enc

    def normalize_increment(self, increment):
        increment_enc = update_and_denormalize(increment, self.inc_mean, self.inc_std, 0, self.alpha, self.training,
                                               self.online_adaptation)
        return increment_enc

    def normalize_context_increment(self, increment):
        increment_enc = update_and_normalize(increment, self.inc_mean, self.inc_std, 0, self.alpha, self.training,
                                             self.online_adaptation)
        return increment_enc

    def normalize_backwards_increment(self, increment):
        increment_enc = update_and_denormalize(increment, self.back_inc_mean, self.back_inc_std, 0, self.alpha,
                                               self.training,
                                               self.online_adaptation)
        return increment_enc

    def encoding_and_normalization_Xt(self, target_x):
        target_x_enc = self.sin_cos_angle_encoding(target_x)
        target_x_enc = update_and_normalize(target_x_enc, self.inp_mean, self.inp_std, 2, self.alpha, self.training,
                                            self.online_adaptation)
        return target_x_enc

    def encoding_and_normalization_Yt(self, target_y):
        target_y_enc = self.sin_cos_angle_encoding(target_y)
        target_y_enc = update_and_normalize(target_y_enc, self.inp_mean, self.inp_std, 3, self.alpha, self.training,
                                            self.online_adaptation)
        return target_y_enc

    @classmethod
    def sin_cos_angle_encoding(cls, state: Float[torch.Tensor, "batch context state"]) -> \
            Float[torch.Tensor, "batch context state"]:
        """
        Encoded the angle (in rad) in the state using sin and cos (two channels)
        Expects the angle to be in the third dimension

        Args:
            state: State to be encoded; shape(batch_size, context_size, state_size)

        As in https://github.com/younggyoseo/CaDM/blob/master/cadm/envs/half_cheetah_env.py
        we emcode the main cheetah angle using sin and cos
        """
        sin_cos_encoding = torch.cat([torch.sin(state[:, :, 2:3]), torch.cos(state[:, :, 2:3])], dim=-1)
        encoded_data = torch.cat([state[:, :, :2], sin_cos_encoding, state[:, :, 3:]], dim=2)

        # Remove the first dimension (the x position) because the half cheetah
        # dynamics are independent of the x pos
        return encoded_data[:, :, 1:]


class AntNormalizerAndEncoder(HalfCheethaNormalizerAndEncoder):

    def __init__(self, inp_size, inc_size, alpha=0.001, online_adaptation: bool = False):
        super().__init__(inp_size, inc_size, alpha, online_adaptation)
        self.extra_dims = -2  #- 2 pos

        self.alpha = alpha
        self.online_adaptation = online_adaptation

    @classmethod
    def sin_cos_angle_encoding(cls, state: Float[torch.Tensor, "batch context state"]) -> \
            Float[torch.Tensor, "batch context state"]:
        """
        Encoded the angle (in rad) in the state using sin and cos (two channels)
        Expects the angle to be in the third dimension

        Args:
            state: State to be encoded; shape(batch_size, context_size, state_size)

        """
        #sin_cos_encoding = torch.cat([torch.sin(state[:, :, 3:7]), torch.cos(state[:, :, 3:7])], dim=-1)
        #encoded_data = torch.cat([state[:, :, :3], sin_cos_encoding, state[:, :, 7:]], dim=2)
        encoded_data = state
        # Remove the first two dimensions (the x, y position) because the ant
        # dynamics are independent of the x,y pos
        return encoded_data[:, :, 2:]


class HopperNormalizerAndEncoder(HalfCheethaNormalizerAndEncoder):

    def __init__(self, inp_size, inc_size, alpha=0.001, online_adaptation: bool = False):
        super().__init__(inp_size, inc_size, alpha, online_adaptation)
        self.extra_dims = 0  #- 1 pos + 1 angle

        self.alpha = alpha
        self.online_adaptation = online_adaptation

    @classmethod
    def sin_cos_angle_encoding(cls, state: Float[torch.Tensor, "batch context state"]) -> \
            Float[torch.Tensor, "batch context state"]:
        """
        Encoded the angle (in rad) in the state using sin and cos (two channels)
        Expects the angle to be in the third dimension

        Args:
            state: State to be encoded; shape(batch_size, context_size, state_size)

        """
        sin_cos_encoding = torch.cat([torch.sin(state[:, :, 2:3]), torch.cos(state[:, :, 2:3])], dim=-1)
        encoded_data = torch.cat([state[:, :, :2], sin_cos_encoding, state[:, :, 3:]], dim=2)
        # Remove the first two dimensions (the x, y position) because the ant
        # dynamics are independent of the x,y pos
        return encoded_data[:, :, 1:]

class RLInvPNormalizerAndEncoder(HalfCheethaNormalizerAndEncoder):

    def __init__(self, inp_size, inc_size, alpha=0.001, online_adaptation: bool = False):
        super().__init__(inp_size, inc_size, alpha, online_adaptation)
        self.extra_dims = 1

    @classmethod
    def sin_cos_angle_encoding(cls, state: Float[torch.Tensor, "batch context state"]) -> \
            Float[torch.Tensor, "batch context state+1"]:
        """
        Encoded the angle (in rad) in the state using sin and cos (two channels)
        Expects the angle to be in the third dimension

        Args:
            state: State to be encoded; shape(batch_size, context_size, state_size)

        https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
        Used in https://academic.oup.com/bioinformatics/article/33/18/2842/3738544?login=false#118817095
        """
        sin_cos_encoding = torch.cat([torch.sin(state[:, :, 1, None]), torch.cos(state[:, :, 1, None])], dim=-1)
        encoded_data = torch.cat([state[:, :, :1], sin_cos_encoding, state[:, :, 2:]], dim=2)

        return encoded_data


class VdpNormalizerAndEncoder(BaseNormalizerAndEncoder):

    def __init__(self, inp_std, inp_mean, inc_std, inc_mean):
        super().__init__()
        self.inp_std = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inp_std])
        self.inp_mean = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inp_mean])
        self.inc_std = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inc_std])
        self.inc_mean = ParameterList([Parameter(torch.Tensor(tmp), requires_grad=False) for tmp in inc_mean])
        self.extra_dims = 0

    def encoding_and_normalization_XcYcXt(self, context_x, context_y, target_x):
        context_x_enc = (context_x - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y - self.inp_mean[1]) / self.inp_std[1]
        target_x_enc = (target_x - self.inp_mean[2]) / self.inp_std[2]
        return context_x_enc, context_y_enc, target_x_enc

    def encoding_and_normalization_XcYcXtYt(self, context_x, context_y, target_x, target_y):
        context_x_enc = (context_x - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y - self.inp_mean[1]) / self.inp_std[1]
        target_x_enc = (target_x - self.inp_mean[2]) / self.inp_std[2]
        target_y_enc = (target_y - self.inp_mean[3]) / self.inp_std[3]
        return context_x_enc, context_y_enc, target_x_enc, target_y_enc

    def encoding_and_normalization_XcYc(self, context_x, context_y):
        context_x_enc = (context_x - self.inp_mean[0]) / self.inp_std[0]
        context_y_enc = (context_y - self.inp_mean[1]) / self.inp_std[1]
        return context_x_enc, context_y_enc

    def normalize_increment(self, increment):
        increment_enc = (increment * self.inc_std[0]) + self.inc_mean[0]
        return increment_enc

    def normalize_output(self, out):
        out_enc = (out * self.inc_std[0]) + self.inc_mean[0]
        return out_enc
