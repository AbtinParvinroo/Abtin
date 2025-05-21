import numpy as np
import matplotlib.pyplot as plt

def rgb2hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-10
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
    s = np.zeros_like(maxc)
    s[maxc != 0] = delta[maxc != 0] / maxc[maxc != 0]
    v = maxc
    hsv = np.stack([h, s, v], axis=-1)
    return hsv
# 👇 فقط از matplotlib برای خوندن عکس استفاده می‌کنیم
img_rgb = plt.imread("your_image.jpg").astype(np.float32)
if img_rgb.max() > 1.0:
    img_rgb = img_rgb / 255.0
img_hsv = rgb2hsv(img_rgb)
# 🎨 رنگ‌های هدف در HSV
target_colors_hsv = np.array([
    [0.0, 1.0, 1.0],
    [30/360, 1.0, 1.0],
    [60/360, 1.0, 1.0],
    [120/360, 1.0, 1.0],
    [240/360, 1.0, 1.0],
    [275/360, 1.0, 1.0],
    [300/360, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

# 🔄 پیدا کردن نزدیک‌ترین رنگ
flat_img = img_hsv.reshape(-1, 3)
distances = np.linalg.norm(flat_img[:, None] - target_colors_hsv[None, :], axis=2)
labels = np.argmin(distances, axis=1)
label_img = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])