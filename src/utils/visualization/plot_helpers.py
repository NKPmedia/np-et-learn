import os
from typing import List, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor


def rotate_pos(pos, angle):
    """ Transformation the coordinate in the angle

    Args:
        pos (numpy.ndarray): local state, shape(data_size, 2)
        angle (float): rotate angle, in radians
    Returns:
        rotated_pos (numpy.ndarray): shape(data_size, 2)
    """
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

    return np.dot(pos, rot_mat.T)


def circle(center_x, center_y, radius, start=0., end=2 * np.pi, n_point=100):
    """ Create circle matrix

    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        start (float): start angle
        end (float): end angle
    Returns:
        circle x : numpy.ndarray
        circle y : numpy.ndarray
    """
    diff = end - start

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i * diff / n_point + start))
        circle_ys.append(center_y + radius * np.sin(i * diff / n_point + start))

    return np.array(circle_xs), np.array(circle_ys)


def circle_with_angle(center_x, center_y, radius, angle):
    """ Create circle matrix with angle line matrix

    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        angle (float): in radians
    Returns:
        circle_x (numpy.ndarray): x data of circle
        circle_y (numpy.ndarray): y data of circle
        angle_x (numpy.ndarray): x data of circle angle
        angle_y (numpy.ndarray): y data of circle angle
    """
    circle_x, circle_y = circle(center_x, center_y, radius)

    angle_x = np.array([center_x, center_x + np.cos(angle) * radius])
    angle_y = np.array([center_y, center_y + np.sin(angle) * radius])

    return circle_x, circle_y, angle_x, angle_y


def square(center_x, center_y, shape, angle):
    """ Create square

    Args:
        center_x (float): the center x position of the square
        center_y (float): the center y position of the square
        shape (tuple): the square's shape(width/2, height/2)
        angle (float): in radians
    Returns:
        square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
        square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
    """
    # start with the up right points
    # create point in counterclockwise, local
    square_xy = np.array([[shape[0], shape[1]],
                          [-shape[0], shape[1]],
                          [-shape[0], -shape[1]],
                          [shape[0], -shape[1]],
                          [shape[0], shape[1]]])
    # translate position to world
    # rotation
    trans_points = rotate_pos(square_xy, angle)
    # translation
    trans_points += np.array([center_x, center_y])

    return trans_points[:, 0], trans_points[:, 1]


def square_with_angle(center_x, center_y, shape, angle):
    """ Create square with angle line

    Args:
        center_x (float): the center x position of the square
        center_y (float): the center y position of the square
        shape (tuple): the square's shape(width/2, height/2)
        angle (float): in radians
    Returns:
        square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
        square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
        angle_x (numpy.ndarray): x data of square angle
        angle_y (numpy.ndarray): y data of square angle
    """
    square_x, square_y = square(center_x, center_y, shape, angle)

    angle_x = np.array([center_x, center_x + np.cos(angle) * shape[0]])
    angle_y = np.array([center_y, center_y + np.sin(angle) * shape[1]])

    return square_x, square_y, angle_x, angle_y


def plot_with_std(axs: Axes,
                  x: Float[Tensor, "data"],
                  y: Float[Tensor, "batch data"],
                  alpha: float = 0.2,
                  label: str = "",
                  plot_std: bool = True,
                  move_avg: int = 1,
                  x_axis_limits: Sequence[float] = None,
                  y_axis_limits: Sequence[float] = None,
                  *args, **kwargs):
    """
    Wrapper around matplotlib.pyplot.plot on an Axes object
    Plot the mean and std of the data over the batch dimension
    Uses the same color for the mean and standard deviation
    Args:
        plot_std: whether to plot the standard deviation
        label:
        alpha: transparency of the standard deviation
        axs: Axes object to plot on
        x: shape(data_size)
        y: shape(batch, data_size)
        *args:
        **kwargs:
    """
    y_avg = torch.nn.functional.avg_pool1d(y, kernel_size=move_avg, stride=1)
    x_avg = x[move_avg - 1:]
    mean = torch.mean(y_avg, dim=0)
    var_val = torch.var(y_avg, dim=0)
    std_val = torch.sqrt(var_val)
    if x_axis_limits is not None:
        axs.set_xlim(x_axis_limits[0], x_axis_limits[1])
    if y_axis_limits is not None:
        axs.set_ylim(y_axis_limits[0], y_axis_limits[1])
    axs.plot(x_avg, mean, label=label, *args, **kwargs)
    if plot_std:
        axs.fill_between(x_avg, mean - std_val, mean + std_val, alpha=alpha, color=axs.lines[-1].get_color())

def plot_channel_subplot(x: Float[Tensor, "time"],
                         y: Union[Float[Tensor, "runs batch time channel"], Sequence[Float[Tensor, "batch time channel"]]],
                         labels: Sequence[str],
                         subtitles: Sequence[str],
                         number_plot_cols: int = 2,
                         std: bool = True,
                         move_avg: int = 1,
                         x_axis_limits: Sequence[float] = None,
                         y_axis_limits: Sequence[float] = None,
                         **kwargs
                         ) -> Figure:
        channel = y[0].size(-1)
        rows = int(np.ceil(channel / number_plot_cols))
        fig = plt.figure(figsize=(6 * number_plot_cols, rows * 3))
        for i in range(channel):
            ax = fig.add_subplot(rows, number_plot_cols, i + 1)
            ax.set_title(subtitles[i])
            for j in range(len(y)):
                if i == 0:
                    plot_with_std(fig.gca(), x, y[j][:, :, i], label=labels[j], plot_std=std, move_avg=move_avg, y_axis_limits=y_axis_limits, x_axis_limits=x_axis_limits, **kwargs)
                else:
                    plot_with_std(fig.gca(), x, y[j][:, :, i], plot_std=std, move_avg=move_avg, y_axis_limits=y_axis_limits, x_axis_limits=x_axis_limits, **kwargs)
        return fig