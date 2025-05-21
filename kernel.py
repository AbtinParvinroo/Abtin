import numpy as np

def applyKernel(img, kernel):
    kh, kw = kernel.shape
    h, w = img.shape
    output = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img[i: i + kh, j: j + kw]
            output[i, j] = np.sum(region * kernel)
    return output

def applayKernel2Layers(layers, kernel):
    h, w, c = layers.shape
    outputs = []
    for i in range(c):
        conv = applyKernel(layers[:, :, i], kernel)
        outputs.append(conv)
    return np.stack(outputs, axis=-1)
