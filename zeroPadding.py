import numpy as np

def zeroPadding(img, pad):
    paddedImg = np.pad(img, pad_width=2, mode='constant', constant_values=0)
    return paddedImg

# h(j) = h + 2(padding)
# w(i) = h + 2(padding)