from model.LassoHomotopy import LassoHomotopyModel
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 samples, 5 features
true_coef = np.array([1.5, -2.0, 0, 0, 0.5])
y = X @ true_coef + np.random.normal(scale=0.1, size=100)

# Fit the LASSO model
model = LassoHomotopyModel(alpha=0.1)
results = model.fit(X, y)

# Make predictions
predictions = results.predict(X)

# Print coefficients
print("Fitted coefficients:", results.coef_)
print("Intercept:", results.intercept_)