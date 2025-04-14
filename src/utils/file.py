import os

import numpy as np

from src.utils.consts import SUBSETS


def load_txt(base_folder, split=None):
    res = {}

    for subset_name in SUBSETS:
        file_path = os.path.join(base_folder, subset_name, split)
        if os.path.exists(file_path):
            res[subset_name] = np.loadtxt(file_path)

    return res
