import numpy as np
import matplotlib.pyplot as plt

def rgb2hsv(rgb):
    # همون تابع قبلی برای تبدیل RGB → HSV با خروجی در بازه [0,1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-10
    # Hue
    h = np.zeros_like(maxc)
    mask = (delta != 0)
    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (maxc == g) & mask
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (maxc == b) & mask
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    h = h / 6
    h[h < 0] += 1
    h[~mask] = 0
    # Saturation
    s = np.zeros_like(maxc)
    nz = (maxc != 0)
    s[nz] = delta[nz] / maxc[nz]
    # Value
    v = maxc
    return np.stack([h, s, v], axis=-1)

def circular_hsv_distance(img_hsv, target_hsv):
    h1, s1, v1 = img_hsv[:,0,None], img_hsv[:,1,None], img_hsv[:,2,None]
    h2, s2, v2 = target_hsv[None,:,0], target_hsv[None,:,1], target_hsv[None,:,2]
    # فاصله دایره‌ای در Hue
    dh = np.abs(h1 - h2)
    dh = np.minimum(dh, 1 - dh)
    # فاصله خطی در S,V
    ds = s1 - s2
    dv = v1 - v2
    return np.sqrt(dh**2 + ds**2 + dv**2)

def quantize_hue(img_hsv, num_bins=9):
    h = img_hsv[...,0]
    # هر بین به اندازه‌ی 1/num_bins
    bins = np.linspace(0, 1, num_bins+1)
    labels = np.digitize(h, bins) - 1
    # برای حالت h==1 بذاریم روی آخری
    labels[labels == num_bins] = num_bins - 1
    return labels
