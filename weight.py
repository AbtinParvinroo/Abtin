import numpy as np
from scipy.ndimage import sobel
from scipy.optimize import minimize

class HSVResolutionWeight:
    def __init__(self, H0=0.5, sigma=0.1, gamma=2, V_th=0.3):
        self.H0 = H0
        self.sigma = sigma
        self.gamma = gamma
        self.V_th = V_th

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

    def compute_weights(self, H, S, V, alphas, dropout_rate=0.1):
        # dropout ساده روی آلفا ها (غیر صفر بمونن)
        mask = np.random.rand(len(alphas)) > dropout_rate
        alphas = np.array(alphas) * mask
        if alphas.sum() == 0:
            alphas = np.ones_like(alphas)
        alphas = alphas / alphas.sum()
        w_H = self.hue_weight(H)
        w_S = self.saturation_weight(S)
        w_V = self.value_weight(V)
        w_R = self.resolution_weight(V)
        combined_weight = (alphas[0] * w_H + alphas[1] * w_S + alphas[2] * w_V + alphas[3] * w_R)
        # نرمال سازی نهایی
        combined_weight = (combined_weight - np.min(combined_weight)) / (np.max(combined_weight) - np.min(combined_weight) + 1e-8)
        return combined_weight

def objective(alphas, hsv_obj, H, S, V, target=None):
    # نرمال سازی و شرط جمع وزن ها برابر 1
    alphas = np.abs(alphas)
    alphas = alphas / (alphas.sum() + 1e-8)
    W = hsv_obj.compute_weights(H, S, V, alphas, dropout_rate=0)
    # اگر هدف مشخص نباشد، خطا را بر اساس پراکندگی وزن ها میگیریم (مثال ساده)
    if target is None:
        # فرض می‌کنیم دنبال وزنی هستیم که پراکندگی رو کمینه کنه (مثال ساده)
        loss = np.var(W)
    else:
        # اگر هدف داده شده بود، فاصله MSE بگیر
        loss = np.mean((W - target) ** 2)
    return loss

def optimize_alphas(hsv_obj, H, S, V, target=None):
    # شروع با وزن‌های مساوی
    x0 = np.array([0.25, 0.25, 0.25, 0.25])
    # محدودیت جمع آلفا برابر 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # هر وزن بین 0 و 1 باشه
    bounds = [(0, 1)] * 4
    res = minimize(objective, x0, args=(hsv_obj, H, S, V, target), bounds=bounds, constraints=cons, method='SLSQP')
    alphas_opt = np.abs(res.x)
    alphas_opt /= alphas_opt.sum()
    return alphas_opt

def test_system(H, S, V):
    print("Starting optimization with test data...")
    hsv_obj = HSVResolutionWeight()
    # بهینه‌سازی آلفا
    best_alphas = optimize_alphas(hsv_obj, H, S, V)
    print(f"Optimized alphas: {best_alphas}")
    # محاسبه وزن نهایی با Dropout (برای تست پایداری)
    W = hsv_obj.compute_weights(H, S, V, best_alphas, dropout_rate=0.1)
    print(f"Weight stats - min: {W.min()}, max: {W.max()}, mean: {W.mean()}")
    return W, best_alphas
