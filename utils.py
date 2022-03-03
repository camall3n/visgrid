import argparse
import glob
import json
import numpy as np
import os
import random
from sklearn.neighbors import KernelDensity
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

def fit_kde(x, bw=0.03):
    p = KernelDensity(bandwidth=bw, kernel='tophat')
    p.fit(x)
    return p

def MI(x, y):
    xy = np.concatenate([x, y], axis=-1)
    log_pxy = fit_kde(xy).score_samples(xy)
    log_px = fit_kde(x).score_samples(x)
    log_py = fit_kde(y).score_samples(y)
    log_ratio = log_pxy - log_px - log_py
    return np.mean(log_ratio)

def get_good_color(color):
    colorname = color
    colorname = 'gold' if colorname == 'yellow' else colorname
    colorname = 'c' if colorname == 'cyan' else colorname
    colorname = 'm' if colorname == 'magenta' else colorname
    colorname = 'silver' if colorname in ['gray', 'grey'] else colorname
    return colorname
