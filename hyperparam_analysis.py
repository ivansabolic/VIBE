import argparse

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    l_vd, accs, asrs = [], [], []
    for line in lines:
        line_split = line.split(',')
        l_vd.append(float(line_split[0].strip()))
        accs.append(float(line_split[1].strip()))
        asrs.append(float(line_split[2].strip()))

    plt.figsize = (20, 10)
    # give me scatter 2 subplots, one l_vd x axis, acc y axis, one l_vd x axis, asr y axis, last acc vs asr
    fig, axs = plt.subplots(3)
    axs[0].scatter(l_vd, accs)
    axs[0].set_title(f'l_vd vs accuracy (pears={pearsonr(l_vd, accs)[0] :.3f})')
    axs[0].set_xlabel('l_vd')
    axs[0].set_ylabel('accuracy')
    axs[1].scatter(l_vd, asrs)
    axs[1].set_title(f'l_vd vs asr (pears={pearsonr(l_vd, asrs)[0] :.3f})')
    axs[1].set_xlabel('l_vd')
    axs[1].set_ylabel('asr')
    axs[2].scatter(accs, asrs)
    axs[2].set_title(f'accuracy vs asr (pears={pearsonr(accs, asrs)[0] :.3f})')
    axs[2].set_xlabel('accuracy')
    axs[2].set_ylabel('asr')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/hyperparam_analysis.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="hyperparam_analysis")
    args = parser.parse_args()
    main(args)