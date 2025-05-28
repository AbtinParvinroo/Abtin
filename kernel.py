import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def applyKernel(img, kernel):
    img = img.astype(np.float32)
    kh, kw = kernel.shape
    assert (kh, kw) == (3, 3)
    h, w = img.shape
    windows = sliding_window_view(img, (kh, kw))
    output = np.einsum('ijkl,kl->ij', windows, kernel)
    return output.astype(np.float32)

def applyKernel2Layers(layers, kernel):
    h, w, c = layers.shape
    outputs = []
    for i in range(c):
        conv = applyKernel(layers[:, :, i], kernel)
        outputs.append(conv)
    return np.stack(outputs, axis=-1)
