import numpy as np


class LassoHomotopyModel:
    def __init__(self, alpha=0.1, tol=1e-6, max_iter=1000):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.beta = None
        self.active_set = []
        self.signs = []
        self.mu = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None

    def _standardize(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean()

        # Handle zero-variance features
        self.X_std[self.X_std == 0] = 1.0

        X_std = (X - self.X_mean) / self.X_std
        y_centered = y - self.y_mean
        return X_std, y_centered

    def fit(self, X, y):
        X, y = self._standardize(X, y)
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("Input matrix X has no features after standardization")
        self.beta = np.zeros(n_features)
        self.mu = np.zeros_like(y)
        self.active_set = []
        self.signs = []

        corr = X.T @ y
        lambda_max = np.max(np.abs(corr))
        lambda_ = lambda_max

        for _ in range(self.max_iter):
            if lambda_ < self.alpha * lambda_max:
                break

            residuals = y - self.mu
            correlations = X.T @ residuals
            inactive = list(set(range(n_features)) - set(self.active_set))

            if not inactive or np.max(np.abs(correlations[inactive])) < self.tol:
                break

            j = inactive[np.argmax(np.abs(correlations[inactive]))]
            sign = np.sign(correlations[j])

            if j not in self.active_set:
                self.active_set.append(j)
                self.signs.append(sign)

            X_active = X[:, self.active_set]
            signs = np.array(self.signs)
            gram = X_active.T @ X_active

            try:
                inv_gram = np.linalg.inv(gram + 1e-8 * np.eye(len(self.active_set)))
            except np.linalg.LinAlgError:
                inv_gram = np.linalg.pinv(gram)

            equi_vec = inv_gram @ signs
            direction = X_active @ equi_vec

            gamma = np.inf
            c = X.T @ residuals
            d = X.T @ direction
            for k in range(n_features):
                if k in self.active_set:
                    continue
                denominator_plus = 1 - d[k] + 1e-12
                denominator_minus = 1 + d[k] + 1e-12

                temp_plus = (lambda_ - c[k]) / denominator_plus
                if temp_plus > 1e-12 and temp_plus < gamma:
                    gamma = temp_plus

                temp_minus = (lambda_ + c[k]) / denominator_minus
                if temp_minus > 1e-12 and temp_minus < gamma:
                    gamma = temp_minus

            for idx, k in enumerate(self.active_set):
                denominator = equi_vec[idx] + 1e-12
                if np.abs(denominator) < 1e-12:
                    continue

                gamma_k = -self.beta[k] / denominator
                if gamma_k > 1e-12 and gamma_k < gamma:
                    gamma = gamma_k
                    drop_idx = idx

            if gamma < 1e-10 or not np.isfinite(gamma):
                break

            self.mu += gamma * direction
            self.beta[self.active_set] += gamma * equi_vec
            lambda_ -= gamma

            if 'drop_idx' in locals():
                del self.active_set[drop_idx]
                del self.signs[drop_idx]

        self.beta = np.nan_to_num(self.beta)
        return LassoHomotopyResults(self.beta, self.X_mean, self.X_std, self.y_mean)


class LassoHomotopyResults:
    def __init__(self, beta, X_mean, X_std, y_mean):
        self.beta = beta
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean

    def predict(self, X):
        if not np.isfinite(self.beta).all():
            raise ValueError("Model contains invalid coefficients - failed to converge")

        X_scaled = (X - self.X_mean) / self.X_std
        return X_scaled @ self.beta + self.y_mean
