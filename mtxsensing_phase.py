#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

def setup_matplotlib():
    # when none, svg.fonttype will fall back to the default font used when
    # embedding the image in LaTeX
    plt.rcParams['svg.fonttype'] = 'none'


def phase_plot(xnum, ynum, save_name, noheader, *paths):
    nplots = len(paths)
    values = np.zeros((nplots, xnum * ynum))
    for i in range(nplots):
        df = np.loadtxt(paths[i], delimiter=',',
                        skiprows=(0 if noheader else 1))
        values[i, :] = df[:, 2]

    # setup indices, labels
    xidx, yidx = np.arange(xnum), np.arange(ynum)
    ylabels = yidx + 1
    xlabels = [str(i * 0.02) if idx % 5 == 0 else '' for idx, i in enumerate(xidx)]
    xlabels[-1] = str(xidx[-1] * 0.02)  # mark last x-value

    # setup figure and save to .svg file
    fig, axes = plt.subplots(nrows=3)
    for i in range(nplots):
        im = axes[i].imshow(np.reshape(values[i, :], (ynum, xnum)),
                       cmap='gray', interpolation='nearest')
    plt.setp(axes, xticks=xidx, yticks=yidx,
             xticklabels=xlabels, yticklabels=ylabels)
    axes[0].set_title(r"\( d = 100, r = 1 \)")
    axes[1].set_title(r"\( d = 100, r = 5 \)")
    axes[2].set_title(r"\( d = 100, r = 10 \)")
    fig.text(0.5, 0.0, "Corruption level", ha="center", va="center")
    fig.text(0.25, 0.5, r"\( m / 2d \)", ha="center", va="center",
             rotation="vertical")

    fig.tight_layout()  # needed for adjusting margins
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    # bbox_inches = 'tight' will remove margins
    plt.savefig(save_name, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Generates a phase transition plot given the phase transition data in a
        sequence of .CSV files for the bilinear / quadratic sensing
        problems.""")
    parser.add_argument('-i','--in_files',
        nargs='+',
        help='A list of input files',
        type=str,
        required=True)
    parser.add_argument("--out_file", '-o',
        type=str,
        help="The path of the output file")
    parser.add_argument("--xnum", '-x',
        type=int,
        help="The number of x-ticks",
        default=25)
    parser.add_argument("--ynum", '-y',
        type=int,
        help="The number of y-ticks",
        default=8)
    parser.add_argument("--noheader", action='store_true',
        help="""Set to include the first row of the .csv files, normally
        reserved for header""")
    args = vars(parser.parse_args())
    in_paths, out_path = args["in_files"], args["out_file"]
    xnum, ynum = args["xnum"], args["ynum"]
    noheader = args["noheader"]
    setup_matplotlib()
    phase_plot(xnum, ynum, out_path, noheader, *in_paths)
