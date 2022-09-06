import argparse

import numpy as np
import matplotlib.colors as colors

def manhattan_dist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x2 - x1) + np.abs(y2 - y1)

def get_parser():
    """Return a nicely formatted argument parser

    This function is a simple wrapper for the argument parser I like to use,
    which has a stupidly long argument that I always forget.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def standardize_colorname(colorname: str):
    standard_colors = {
        'yellow': 'gold',
        'cyan': 'c',
        'magenta': 'm',
        'orange': 'xkcd:orange',
        'gray': 'silver',
        'grey': 'silver',
        'almost black': 'xkcd:almost black',
    }
    if colorname in standard_colors:
        colorname = standard_colors[colorname]
    return colorname

def get_rgb(colorname: str):
    good_color = standardize_colorname(colorname)
    color_tuple = colors.hex2color(colors.get_named_colors_mapping()[good_color])
    return np.asarray(color_tuple)

def to_rgb(array: np.ndarray, color=None):
    """Add a channel dimension with 3 entries"""
    if array.ndim == 3 and array.shape[-1] == 3:
        pass
    else:
        array = np.tile(array[:, :, np.newaxis], (1, 1, 3))
    array = array.astype(float)
    if color is not None:
        array *= get_rgb(color)
    return array
