import numpy as np

class DynamicLossTrainer:
    def __init__(self, X, y_true, lr_model=0.01, lr_alpha=0.01):
        self.X = X
        self.y_true = y_true
        self.w_model = np.random.randn(X.shape[1])
        self.logits_alpha = np.zeros(2)
        self.lr_model = lr_model
        self.lr_alpha = lr_alpha

    def softMax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def model(self):
        return self.X @ self.w_model

    def mseLoss(self, y_pred):
        return np.mean((y_pred - self.y_true))

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

    def train_step(self):
        y_pred = self.model()
        loss1 = self.mse_loss(y_pred)
        loss2 = self.mae_loss(y_pred)
        alpha = self.softmax(self.logits_alpha)
        total_loss = alpha[0]*loss1 + alpha[1]*loss2
        grad_w_model = alpha[0]*self.grad_mse(y_pred) + alpha[1]*self.grad_mae(y_pred)
        grad_alpha_logits = np.array([loss1, loss2]) - total_loss
        self.w_model -= self.lr_model * grad_w_model
        self.logits_alpha -= self.lr_alpha * grad_alpha_logits
        return total_loss, alpha

    def train(self, epochs=100, verbose=10):
        for epoch in range(epochs):
            loss, alpha = self.train_step()
            if epoch % verbose == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Alphas={alpha}")