import colorsys
import glob
import json
import os
import random
import string
import subprocess
from typing import List
import matplotlib.colors as mc
import imageio
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def show_gif(fname: str):
    import base64
    from IPython import display
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')  #


def videofy_figures(figures: List[Figure], name: str = "video_name.mp4"):
    for i in range(len(figures)):
        figures[i].savefig("/tmp/video/file%02d.png" % i)

    os.chdir("/tmp/video")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        name
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)


def giffify_figures(figures: List[Figure], duration: float = 4000):
    """
    Creates a gif from a list of figures.
    Saves the gif in /tmp and returns the path to the gif.
    Args:
        figures:

    Returns:
        Path to the gif.
    """
    png_files = []
    for i, fig in enumerate(figures):
        png_file = f'/tmp/img_{i}.png'
        png_files.append(png_file)
        fig.savefig(png_file,
                    transparent=False,
                    facecolor='white'
                    )
    gifname = giffify_pngs(png_files, duration=duration)
    for i, fig in enumerate(figures):
        os.remove(f'/tmp/img_{i}.png')
    return gifname


def giffify_pngs(png_files: List[str], duration: float = 4000) -> str:
    gifname = ''.join(random.choices(string.ascii_lowercase, k=5))
    gife_name = f"/tmp/{gifname}.gif"
    frames = []
    for file in png_files:
        frame = imageio.v2.imread(file)
        frames.append(frame)
    imageio.mimsave(gife_name,
                    frames,
                    duration=duration, loop=0)
    return gife_name


def set_size(width_pt: int = None, height_pt: int = None, fraction=1, subplots=(1, 1), ratio: float = None):
    """
    Set figure dimensions to sit nicely in our document.
    Use `\the\textwidth` in your latex document to get the valid width in pts.

    Normal beamer frame width is 398.
    Normal paper width is 398.

    Normal beamer frame height is 265.

    Parameters
    ----------
    width_pt: float
        Document width in points
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
        The number of rows and columns of subplots.

    Returns
    -------
    tuple
        Dimensions of figure in inches

    Examples
    -------
    >>> fig_x, fig_y = set_size(398, subplots=(1, 2))
    """
    assert (width_pt is not None or height_pt is not None) and (width_pt is None or height_pt is None), \
        "Either width_pt or height_pt must be specified, but not both."

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if ratio is None:
        golden_ratio = (5 ** .5 - 1) / 2
        ratio = golden_ratio

    # Width of figure (in pts)
    if width_pt is not None:
        fig_width_pt = width_pt * fraction
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

        # Add space for x labels
        fig_height_in += 50 * inches_per_pt

    if height_pt is not None:
        fig_height_pt = height_pt * fraction
        # Figure height in inches
        fig_height_in = fig_height_pt * inches_per_pt
        # Figure width in inches
        fig_width_in = fig_height_in / ratio * (subplots[1] / subplots[0])

    return (fig_width_in, fig_height_in)


def load_colors():
    """Get the official RWTH Aachen University colors.

    Parameters
    ----------
    Returns
    -------
    dict
        Dictionary with all the RWTH Aachen University colors.
    """
    # get location of this file
    path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(path)
    color_json = glob.glob(parent_path + '/**/RWTHcolors.json', recursive=True)
    with open(color_json[0]) as json_file:
        c = json.load(json_file)

    import matplotlib as mpl

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        c["blau100"],
        c["orange100"],
        c["gruen100"],
        c["rot100"],
        c["violett100"],
        c["schwarz50"],
        c["tuerkis100"],
        c["petrol100"],
        c["yellow100"],
    ])
    return c


def initialize_plot(conference):
    """Get parameters for matplotlib.

    Parameters
    ----------
    conference: str
        Name of the conference for default values. (So far: {'AAAI', 'CDC'})

    Returns
    -------
    dict:
        Dictionary with the parameters

    Examples
    -------
    >>> import matplotlib as mpl
    >>> params = initialize_plot('AAAI')
    >>> mpl.rcParams.update(params)
    """

    if conference == 'AAAI':
        plot_params = {
            "font.family": "serif",
            "text.usetex": True,
            'text.latex.preamble': [r'\usepackage{amsmath}',
                                    r'\usepackage{amssymb}']
        }
    elif conference == 'CDC':
        plot_params = {
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': True,
            'axes.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.labelsize': 9,
            "pgf.preamble": "\n".join([
                r'\usepackage{bm}',
            ]),
            'text.latex.preamble': [r'\usepackage{amsmath}',
                                    r'\usepackage{amssymb}',
                                    r'\usepackage{bm}'],
        }
    elif conference == 'README':
        plot_params = {
            'font.family': 'serif',
            'text.usetex': True,
            'axes.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.labelsize': 10,
            "legend.fontsize": 10,
            "pgf.preamble": "\n".join([
                r'\usepackage{bm}',
            ]),
            'text.latex.preamble': "\n".join([r'\usepackage{amsmath}',
                                              r'\usepackage{amssymb}',
                                              r'\usepackage{bm}']),
        }
    else:
        plot_params = {}

    return plot_params


def lighten_color(color, amount=0.5):
    """Lighten a given color. Note that the result will no longer be compliant
    with the RWTH Aachen University color formatting.

    Parameters
    ----------
    color
        Color as matplotlib color string, hex string, or RGB tuple
    amount: float, default=0.5
        Amount by which the given color is multiplied (1-luminosity).

    Returns
    -------
    tuple
        RGB tuple with the lighter color.

    Examples
    --------
    >>> lighten_color('g', 0.3)
    >>> lighten_color('#F034A3', 0.6)
    >>> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
