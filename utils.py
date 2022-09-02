import argparse
import glob
import json
import numpy as np
import os
import random

import torch

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

def get_good_color(color):
    colorname = color
    colorname = 'gold' if colorname == 'yellow' else colorname
    colorname = 'c' if colorname == 'cyan' else colorname
    colorname = 'm' if colorname == 'magenta' else colorname
    colorname = 'silver' if colorname in ['gray', 'grey'] else colorname
    return colorname
