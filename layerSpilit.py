import numpy as np

def layerSpilit(paddedLabelImg, numClasses=9):
    H, W = paddedLabelImg.shape
    layers = np.zeros((H, W, numClasses), dtype=np.float32)
    for c in range(numClasses):
        layers[:, :, c] = (paddedLabelImg == c).astype(np.float32)
    return layers
