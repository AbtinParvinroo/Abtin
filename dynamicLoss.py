import numpy as np

class DynamicLossTrainer:
    def __init__(self, X, y_true, lr_model=0.01, lr_alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.X = X
        self.y_true = y_true
        self.w_model = np.random.randn(X.shape[1])
        self.logits_alpha = np.zeros(2)
        self.lr_model = lr_model
        self.lr_alpha = lr_alpha
        # Adam optimizer state for w_model
        self.m_w = np.zeros_like(self.w_model)
        self.v_w = np.zeros_like(self.w_model)
        # Adam optimizer state for logits_alpha
        self.m_alpha = np.zeros_like(self.logits_alpha)
        self.v_alpha = np.zeros_like(self.logits_alpha)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def model(self):
        return self.X @ self.w_model

    def mseLoss(self, y_pred):
        return np.mean((y_pred - self.y_true) ** 2)

    def grad_mse(self, y_pred):
        N = self.y_true.shape[0]
        grad_y = 2 * (y_pred - self.y_true) / N
        grad_w = self.X.T @ grad_y
        return grad_w

    def mae_loss(self, y_pred):
        return np.mean(np.abs(y_pred - self.y_true))

    def grad_mae(self, y_pred):
        N = self.y_true.shape[0]
        grad_y = np.where(y_pred > self.y_true, 1, -1) / N
        grad_w = self.X.T @ grad_y
        return grad_w

    def normalize_losses(self, losses):
        # Normalize losses to zero mean, unit variance for stability
        losses = np.array(losses)
        mean = losses.mean()
        std = losses.std() + 1e-8
        return (losses - mean) / std

    def grad_alpha(self, losses, alpha):
        # Correct gradient of logits_alpha using softmax derivative
        grad = np.zeros_like(alpha)
        L = np.dot(alpha, losses)
        for i in range(len(alpha)):
            s = 0
            for j in range(len(alpha)):
                delta = 1 if i == j else 0
                s += alpha[j] * (delta - alpha[i]) * (losses[j] - L)
            grad[i] = s
        return grad

    def adam_update(self, param, grad, m, v, lr):
        self.t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param_update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
        param -= param_update
        return param, m, v

    def train_step(self):
        y_pred = self.model()
        loss1 = self.mseLoss(y_pred)
        loss2 = self.mae_loss(y_pred)
        # Normalize losses for stable gradient on alphas
        norm_losses = self.normalize_losses([loss1, loss2])
        alpha = self.softmax(self.logits_alpha)
        total_loss = alpha[0]*loss1 + alpha[1]*loss2
        grad_w_model = alpha[0]*self.grad_mse(y_pred) + alpha[1]*self.grad_mae(y_pred)
        grad_alpha_logits = self.grad_alpha(norm_losses, alpha)
        # Update weights and logits_alpha with Adam
        self.w_model, self.m_w, self.v_w = self.adam_update(self.w_model, grad_w_model, self.m_w, self.v_w, self.lr_model)
        self.logits_alpha, self.m_alpha, self.v_alpha = self.adam_update(self.logits_alpha, grad_alpha_logits, self.m_alpha, self.v_alpha, self.lr_alpha)
        return total_loss, alpha

    def train(self, epochs=100, verbose=10):
        for epoch in range(epochs):
            loss, alpha = self.train_step()
            if epoch % verbose == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Alphas={alpha}")
