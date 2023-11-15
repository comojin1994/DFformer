from typing import Iterable
import torch
import torchaudio
import numpy as np


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)


class MinMaxNormalization:
    def __init__(self):
        pass

    def __call__(self, x):
        x = (x - x.min()) / (x.max() - x.min())

        return x


class ChannelPermutation:
    def __init__(self):
        pass

    def __call__(self, x):
        permuted_idx = np.random.permutation(x.shape[0])
        permuted_data = x[permuted_idx, :, :]
        return permuted_data
