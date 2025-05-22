import numpy as np
from scipy.ndimage import sobel

class HSVResolutionWeight:
    def __init__(self, H0=0.5, sigma=0.1, gamma=2, V_th=0.3,
                    alpha_H=0.25, alpha_S=0.25, alpha_V=0.25, alpha_R=0.25):
        self.H0 = H0
        self.sigma = sigma
        self.gamma = gamma
        self.V_th = V_th
        self.alpha_H = alpha_H
        self.alpha_S = alpha_S
        self.alpha_V = alpha_V
        self.alpha_R = alpha_R

    @staticmethod
    def circular_distance(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 1 - diff)

    def hue_weight(self, H):
        d = self.circular_distance(H, self.H0)
        w = np.exp(- (d ** 2) / (2 * self.sigma**2))
        return w

    def saturation_weight(self, S):
        return S ** self.gamma

    def value_weight(self, V):
        return 1 / (1 + np.exp(-10 * (V - self.V_th)))

    def resolution_weight(self, V):
        dx = sobel(V, axis=0, mode='reflect')
        dy = sobel(V, axis=1, mode='reflect')
        grad_mag = np.sqrt(dx**2 + dy**2)
        grad_norm = grad_mag / (np.max(grad_mag) + 1e-8)
        return grad_norm

    def compute(self, H, S, V):
        w_H = self.hue_weight(H)
        w_S = self.saturation_weight(S)
        w_V = self.value_weight(V)
        w_R = self.resolution_weight(V)
        W = (self.alpha_H * w_H + self.alpha_S * w_S + self.alpha_V * w_V + self.alpha_R * w_R)
        W = (W - np.min(W)) / (np.max(W) - np.min(W) + 1e-8)
        return W