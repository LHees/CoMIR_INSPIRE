import sys
import os
import glob

import skimage.io as skio
import numpy as np
from scipy.stats import pearsonr


def evaluate(path, mod1, mod2):
    comirs1 = [skio.imread(sub, plugin='simpleitk') for sub in
               glob.glob(os.path.join(path, mod1+"/*"))]
    comirs2 = [skio.imread(sub, plugin='simpleitk') for sub in
               glob.glob(os.path.join(path, mod2+"/*"))]

    flat_comirs1 = [comir.flatten() for comir in comirs1]
    flat_comirs2 = [comir.flatten() for comir in comirs2]

    return [pearsonr(flat1, flat2).statistic for flat1, flat2 in
            zip(flat_comirs1, flat_comirs2)]


if __name__ == "__main__":
    if len(sys.argv) in [1, 3] or len(sys.argv) > 4:
        raise ValueError("Usage: python evaluate_comir.py path_to_comirs "
                         "[modality1 modality2]")
    path = sys.argv[1]
    mod1 = "T1w"
    mod2 = "flair"
    if len(sys.argv) == 4:
        mod1, mod2 = sys.argv[2:]
    rs = evaluate(path, mod1, mod2)
    print(rs)
    print(f'mean: {np.mean(rs)}')
