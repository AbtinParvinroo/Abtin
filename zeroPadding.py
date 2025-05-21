import numpy as np

def reflectPadding(img, pad):
    paddedImg = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    return paddedImg

# h(j) = h + 2(padding)
# w(i) = h + 2(padding)