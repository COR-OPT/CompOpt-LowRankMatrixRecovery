#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

def setup_matplotlib():
    # when none, svg.fonttype will fall back to the default font used when
    # embedding the image in LaTeX
    plt.rcParams['svg.fonttype'] = 'none'


def phase_plot(path, save_name, noheader):
    xnum, ynum = 40, 6
    nplots = 1
    values = np.zeros((nplots, xnum * ynum))
    # load data from file
    df = np.loadtxt(path, delimiter=",", skiprows=(0 if noheader else 1))
    values[0, :] = df[:, 2]

    # setup indices, labels
    xidx, yidx = np.arange(xnum), np.arange(ynum)
    ylabels = [1, 2, 4, 6, 8, 10]
    xlabels = [str(i * 0.025) if idx % 10 == 0 else '' for idx, i in enumerate(xidx)]
    xlabels[-1] = ('%.3f' % (xidx[-1] * 0.025))  # mark last x-value

    # setup figure and save to .svg file
    plt.imshow(np.reshape(values[0, :], (ynum, xnum)), cmap='gray',
            interpolation='nearest')
    plt.xticks(xidx, xlabels)
    plt.yticks(yidx, ylabels)
    plt.title(r"\( d = 100 \)")
    plt.xlabel(r"\( \mathbb{P}(\delta_{ij} = 1) \)", labelpad=2)
    plt.ylabel(r"Rank \( r \)", labelpad=2, rotation="vertical")

    plt.tight_layout()  # needed for adjusting margins
    # fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    # bbox_inches = 'tight' will remove margins
    plt.savefig(save_name, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Generates a phase transition plot given the phase transition data in a
        sequence of .CSV files for the low-rank matrix completion problem.""")
    parser.add_argument('-i','--in_file',
        help='The path of the input file',
        type=str,
        required=True)
    parser.add_argument("--out_file", '-o',
        type=str,
        help="The path of the output file")
    parser.add_argument("--noheader", action='store_true',
        help="""Set to include the first row of the .csv files, normally
        reserved for header""")
    args = vars(parser.parse_args())
    in_path, out_path = args["in_file"], args["out_file"]
    noheader = args["noheader"]
    setup_matplotlib()
    phase_plot(in_path, out_path, noheader)
