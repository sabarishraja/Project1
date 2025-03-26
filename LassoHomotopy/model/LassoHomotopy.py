import numpy as np

class LassoHomotopyModel:
    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000):
        """
        Initialize the LassoHomotopyModel.
        
        Parameters:
        - alpha: Regularization strength (λ).
        - tol: Convergence tolerance.
        - max_iter: Maximum number of iterations for the algorithm.
        """
        self.alpha = alpha  # Regularization parameter
        self.tol = tol      # Tolerance for convergence
        self.max_iter = max_iter  # Maximum iterations
        self.coef_ = None   # Coefficients after fitting
        self.intercept_ = None  # Intercept term (if applicable)

    def fit(self, X, y):
        """
        Fit the LASSO model using the Homotopy Method.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features).
        - y: Target vector (n_samples,).
        
        Returns:
        - An instance of LassoHomotopyResults containing the fitted coefficients.
        """
        n_samples, n_features = X.shape
        X = np.hstack([np.ones((n_samples, 1)), X])  # Add intercept term
        y = y.flatten()  # Ensure y is a 1D array

        # Initialize variables
        beta = np.zeros(n_features + 1)  # Coefficients (including intercept)
        active_set = []  # Indices of active features
        inactive_set = list(range(1, n_features + 1))  # Exclude intercept

        residual = y  # Initial residual
        for iteration in range(self.max_iter):
            # Compute correlation between features and residual
            correlations = X[:, inactive_set].T @ residual
            if len(active_set) > 0:
                correlations_active = X[:, active_set].T @ residual
                correlations = np.concatenate([correlations_active, correlations])

            # Find the feature with the largest correlation
            max_corr_idx = np.argmax(np.abs(correlations))
            if max_corr_idx < len(active_set):
                # Update existing active feature
                update_idx = active_set[max_corr_idx]
            else:
                # Add new feature to active set
                update_idx = inactive_set[max_corr_idx - len(active_set)]
                active_set.append(update_idx)
                inactive_set.remove(update_idx)

            # Solve least squares on active set
            X_active = X[:, active_set]
            beta_active = np.linalg.lstsq(X_active, y, rcond=None)[0]

            # Apply soft thresholding for LASSO
            beta_active = np.sign(beta_active) * np.maximum(np.abs(beta_active) - self.alpha, 0)

            # Update coefficients
            beta[active_set] = beta_active
            residual = y - X @ beta

            # Check for convergence
            if np.linalg.norm(residual) < self.tol:
                break

        # Store the final coefficients
        self.coef_ = beta[1:]  # Exclude intercept
        self.intercept_ = beta[0]

        # Return results object
        return LassoHomotopyResults(self.coef_, self.intercept_)


class LassoHomotopyResults:
    def __init__(self, coef, intercept):
        """
        Initialize the results of the LASSO fit.
        
        Parameters:
        - coef: Fitted coefficients.
        - intercept: Intercept term.
        """
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        """
        Predict using the fitted LASSO model.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features).
        
        Returns:
        - Predicted values (n_samples,).
        """
        return X @ self.coef_ + self.intercept_