"""
GAIN (Yoon, Jordon & van der Schaar, 2018) implemented in pure NumPy.

Architecture follows the reference implementation:
  Generator     : [X_tilde, M] -> FC(h) -> ReLU -> FC(h) -> ReLU -> FC(d) -> Sigmoid
  Discriminator : [X_hat,  H] -> FC(h) -> ReLU -> FC(h) -> ReLU -> FC(d) -> Sigmoid
  Loss_D = -mean[ M log D + (1-M) log(1-D) ]
  Loss_G = -mean_{missing}[ log D ] + alpha * MSE_{observed}(X, G)
Optimiser: Adam. Data are min-max scaled with observed values only.
"""
import numpy as np


class _Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class _MLP:
    """3-layer MLP with ReLU hidden layers and sigmoid output (Xavier init)."""

    def __init__(self, n_in, n_hidden, n_out, rng):
        def xavier(a, b):
            return rng.normal(0, np.sqrt(2.0 / (a + b)), size=(a, b))
        self.W1, self.b1 = xavier(n_in, n_hidden), np.zeros(n_hidden)
        self.W2, self.b2 = xavier(n_hidden, n_hidden), np.zeros(n_hidden)
        self.W3, self.b3 = xavier(n_hidden, n_out), np.zeros(n_out)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        z1 = x @ self.W1 + self.b1; a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3; out = 1.0 / (1.0 + np.exp(-z3))
        self._cache = (x, z1, a1, z2, a2, out)
        return out

    def backward(self, d_out):
        """d_out = dL/d(out). Returns parameter grads and dL/dx."""
        x, z1, a1, z2, a2, out = self._cache
        dz3 = d_out * out * (1 - out)
        gW3, gb3 = a2.T @ dz3, dz3.sum(0)
        da2 = dz3 @ self.W3.T; dz2 = da2 * (z2 > 0)
        gW2, gb2 = a1.T @ dz2, dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        gW1, gb1 = x.T @ dz1, dz1.sum(0)
        dx = dz1 @ self.W1.T
        return [gW1, gb1, gW2, gb2, gW3, gb3], dx


def gain_impute(X, batch_size=128, hint_rate=0.9, alpha=100.0,
                iterations=10000, learning_rate=1e-3, hidden_dim=None,
                random_state=0):
    """Impute NaNs in a 2-D float array with GAIN. Observed cells are kept."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    M = (~np.isnan(X)).astype(float)
    h = hidden_dim or max(32, 2 * d)

    lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
    rng_ = np.where(hi - lo == 0, 1.0, hi - lo)
    Xn = np.nan_to_num((X - lo) / rng_, nan=0.0)

    G = _MLP(2 * d, h, d, rng)
    D = _MLP(2 * d, h, d, rng)
    optG = _Adam(G.params, learning_rate)
    optD = _Adam(D.params, learning_rate)
    eps = 1e-8

    for _ in range(iterations):
        idx = rng.integers(0, n, size=min(batch_size, n))
        x, m = Xn[idx], M[idx]
        b = len(idx)
        z = rng.uniform(0, 0.01, size=(b, d))
        x_tilde = m * x + (1 - m) * z
        hb = (rng.uniform(size=(b, d)) < hint_rate).astype(float)
        hint = m * hb + 0.5 * (1 - hb)

        # ---- Discriminator step ----
        g = G.forward(np.hstack([x_tilde, m]))
        x_hat = m * x + (1 - m) * g
        dp = D.forward(np.hstack([x_hat, hint]))
        d_dp = -(m / (dp + eps) - (1 - m) / (1 - dp + eps)) / (b * d)
        gradsD, _ = D.backward(d_dp)
        optD.step(gradsD)

        # ---- Generator step ----
        g = G.forward(np.hstack([x_tilde, m]))
        x_hat = m * x + (1 - m) * g
        dp = D.forward(np.hstack([x_hat, hint]))
        n_mis = (1 - m).sum() + eps
        n_obs = m.sum() + eps
        d_dp = -(1 - m) / (dp + eps) / n_mis
        _, dx = D.backward(d_dp)
        d_xhat = dx[:, :d]
        d_g = d_xhat * (1 - m) + alpha * (2 * m * (g - x) / n_obs)
        gradsG, _ = G.backward(d_g)
        optG.step(gradsG)

    z = rng.uniform(0, 0.01, size=(n, d))
    x_tilde = M * Xn + (1 - M) * z
    g = G.forward(np.hstack([x_tilde, M]))
    out = g * rng_ + lo
    X_imp = X.copy()
    X_imp[M == 0] = out[M == 0]
    return X_imp
