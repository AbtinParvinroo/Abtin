import numpy as np
from scipy.special import erf
from numpy.lib.stride_tricks import sliding_window_view

class ActivationFunction:
    def __init__(self, beta=1.0, pad=2, kernel_size=3):
        self.beta = beta
        self.pad = pad
        self.kernel_size = kernel_size

    def swish(self, x):
        sigmoid = 1 / (1 + np.exp(-self.beta * x))
        return x * sigmoid

    def gelu(self, x):
        return x * 0.5 * (1 + erf(x / np.sqrt(2)))

    def mark_padding_regions(self, input_shape):
        h, w = input_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        padded_mask = np.pad(mask, self.pad, mode='constant', constant_values=1)
        padded_mask[self.pad: -self.pad, self.pad: -self.pad] = 0
        windowed = sliding_window_view(padded_mask, (self.kernel_size, self.kernel_size))
        marked = (np.sum(windowed, axis=(2, 3)) > 0).astype(np.uint8)
        return marked.astype(bool)

    def apply(self, x):
        padding_mask = self.mark_padding_regions(x.shape)
        out = np.zeros_like(x)
        out[padding_mask] = self.gelu(x[padding_mask])
        out[~padding_mask] = self.swish(x[~padding_mask])
        return out