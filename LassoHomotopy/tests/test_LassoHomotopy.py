import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from Project1.LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel


def generate_data(n_samples=100, n_features=10, noise=0.1, random_state=42):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    return X, y

def test_lasso_homotopy():
    X, y = generate_data()
    model = LassoHomotopyModel()
    sk_model = Lasso(alpha=0.1)
    results = model.fit(X, y)
    sk_model.fit(X, y)
    print("\nTest Case 1: \nBasic Usage - Coefficients:", results.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", sk_model.coef_)


    X_col = np.hstack((X, X[:, :3]))
    res_col = model.fit(X_col, y)
    skl_col = sk_model.fit(X_col, y)
    print("\nTest Case 2: \nCollinear Features - Coefficients:", res_col.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_col.coef_)



    X, y= generate_data(n_samples=100, n_features=50)
    model_high_dim = LassoHomotopyModel()
    res_hd= model_high_dim.fit(X, y)
    skl_hd = sk_model.fit(X, y)
    print("\nTest Case 3: \nHigh-Dimensional Data - Coefficients:", res_hd.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_hd.coef_)


    X, y = generate_data(noise=0.01)
    model_low_noise = LassoHomotopyModel()
    res_ln = model_low_noise.fit(X, y)
    skl_ln = sk_model.fit(X, y)
    print("\nTest Case 4: \nLow Noise - Coefficients:", res_ln.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_ln.coef_)


    X, y = generate_data(noise=1.0)
    model_high_noise = LassoHomotopyModel()
    res_hn = model_high_noise.fit(X, y)
    skl_hn= sk_model.fit(X, y)
    print("\nTest Case 5: \nHigh Noise - Coefficients:", res_hn.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_hn.coef_)


    X_s, y = generate_data(n_samples=100, n_features=20)
    X_s[:, 10:] = 0
    model_sparse = LassoHomotopyModel()
    res_sp = model_sparse.fit(X_s, y)
    skl_sp = sk_model.fit(X_s, y)
    print("\nTest Case 6: \nSparse Data - Coefficients:", res_sp.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_sp.coef_)


    X_n = np.abs(X)
    model_non_negative = LassoHomotopyModel()
    res_nn = model_non_negative.fit(X_n, y)
    skl_nn = sk_model.fit(X_n, y)
    print("\nTest Case 7: \nNon-Negative Features - Coefficients:", res_nn.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_nn.coef_)


    X, y = generate_data(n_samples=1000)
    model_large = LassoHomotopyModel()
    res_l = model_large.fit(X, y)
    skl_l = sk_model.fit(X, y)
    print("\nTest Case 8: \nLarge Dataset - Coefficients:", res_l.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_l.coef_)

    X, y = generate_data(n_samples=20)
    model_small = LassoHomotopyModel()
    res_sm = model_small.fit(X, y)
    skl_sm = sk_model.fit(X, y)
    print("\nTest Case 9: \nSmall Dataset - Coefficients:", res_sm.beta)
    print("\nComparison with Scikit-Learn - Coefficients:", skl_sm.coef_)


    alphas = [0.001, 0.01, 0.1, 1.0]
    for alpha in alphas:
        model_alpha = LassoHomotopyModel(alpha=alpha)
        res_alp = model_alpha.fit(X, y)
        skl_ = Lasso(alpha=alpha)
        skl_alp = skl_.fit(X, y)
        print(f"\nTest Case 10: \nAlpha={alpha} - Coefficients:", res_alp.beta)
        print("\nComparison with Scikit-Learn - Coefficients:", skl_alp.coef_)


    plt.figure(figsize=(12, 10))
    plt.title('Test result page 1')

    plt.subplot(2,2,1)
    plt.plot(results.beta, label='LassoHomotopyModel')
    plt.plot(sk_model.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 1')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(res_col.beta, label='LassoHomotopyModel')
    plt.plot(skl_col.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 2')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(res_hd.beta, label='LassoHomotopyModel')
    plt.plot(skl_hd.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 3')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(res_ln.beta, label='LassoHomotopyModel')
    plt.plot(skl_ln.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 4')
    plt.legend()

    plt.figure(figsize=(12, 10))
    plt.title('Test result page 2')

    plt.subplot(2, 2, 1)
    plt.plot(res_hn.beta, label='LassoHomotopyModel')
    plt.plot(skl_hn.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 5')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(res_sp.beta, label='LassoHomotopyModel')
    plt.plot(skl_sp.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 6')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(res_nn.beta, label='LassoHomotopyModel')
    plt.plot(skl_nn.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 7')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(res_l.beta, label='LassoHomotopyModel')
    plt.plot(skl_l.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 8')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title('Test result page 3')

    plt.subplot(2, 2, 1)
    plt.plot(res_sm.beta, label='LassoHomotopyModel')
    plt.plot(skl_sm.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 9')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(res_alp.beta, label='LassoHomotopyModel')
    plt.plot(skl_alp.coef_, label='Scikit-Learn Lasso')
    plt.title('Coefficient Comparison for test case 10')
    plt.legend()

    plt.tight_layout()
    plt.show()

test_lasso_homotopy()

X, y = generate_data()
model = LassoHomotopyModel()
results = model.fit(X, y)

sk_model = Lasso(alpha=0.1)
sk_model.fit(X, y)

small_test = pd.read_csv('small_test.csv')
collinear_data = pd.read_csv('collinear_data.csv')

#fitting the small_test.csv
X_small = small_test.drop('y', axis=1)
y_small = small_test['y']
X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(X_small, y_small, test_size=0.2, random_state=42)
custom_model_small = LassoHomotopyModel(alpha=0.1)
custom_results_small = custom_model_small.fit(X_small_train.values, y_small_train.values)
sk_model_small = Lasso(alpha=0.1)
sk_model_small.fit(X_small_train, y_small_train)
custom_pred_small  = custom_results_small.predict(X_small_test.values)
sklearn_pred_small = sk_model_small.predict(X_small_test)
custom_mse_small = mean_squared_error(y_small_test, custom_pred_small)
sklearn_mse_small = mean_squared_error(y_small_test, sklearn_pred_small)
custom_r2_small = r2_score(y_small_test, custom_pred_small)
sklearn_r2_small = r2_score(y_small_test, sklearn_pred_small)
print("\nSmall Test Dataset Results:")
print(f"\nCustom Model - MSE: {custom_mse_small:.4f}, R2: {custom_r2_small:.4f}")
print(f"\nSklearn Model - MSE: {sklearn_mse_small:.4f}, R2: {sklearn_r2_small:.4f}")

#fitting the collinear_data.csv
X_collinear = collinear_data.drop('target', axis=1)
y_collinear = collinear_data['target']
X_collinear_train, X_collinear_test, y_collinear_train, y_collinear_test = train_test_split(X_collinear, y_collinear, test_size=0.2, random_state=42)
custom_model_collinear = LassoHomotopyModel(alpha=0.1)
custom_results_collinear = custom_model_collinear.fit(X_collinear_train.values, y_collinear_train.values)
sk_model_collinear = Lasso(alpha=0.1)
sk_model_collinear.fit(X_collinear_train, y_collinear_train)
custom_pred_collinear = custom_results_collinear.predict(X_collinear_test.values)
sk_pred_collinear = sk_model_collinear.predict(X_collinear_test)
custom_mse_collinear = mean_squared_error(y_collinear_test, custom_pred_collinear)
sk_mse_collinear = mean_squared_error(y_collinear_test, sk_pred_collinear)
custom_r2_collinear = r2_score(y_collinear_test, custom_pred_collinear)
sk_r2_collinear = r2_score(y_collinear_test, sk_pred_collinear)
print("\nCollinear Dataset Results:")
print(f"\nCustom Model - MSE: {custom_mse_collinear:.4f}, R2: {custom_r2_collinear:.4f}")
print(f"\nSklearn Model - MSE: {sk_mse_collinear:.4f}, R2: {sk_r2_collinear:.4f}")

#plotting collinear and small_test results
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_small_test, custom_pred_small, alpha=0.5, label='Custom Model')
plt.scatter(y_small_test, sklearn_pred_small, alpha=0.5, label='Sklearn Model')
plt.plot([y_small_test.min(), y_small_test.max()], [y_small_test.min(), y_small_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Small Test Dataset: Actual vs Predicted')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(y_collinear_test, custom_pred_collinear, alpha=0.5, label='Custom Model')
plt.scatter(y_collinear_test, sk_pred_collinear, alpha=0.5, label='Sklearn Model')
plt.plot([y_collinear_test.min(), y_collinear_test.max()], [y_collinear_test.min(), y_collinear_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Collinear Dataset: Actual vs Predicted')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(range(len(custom_results_small.beta)), custom_results_small.beta, alpha=0.5, label='Custom Model')
plt.bar(range(len(sk_model_small.coef_)), sk_model_small.coef_, alpha=0.5, label='Sklearn Model')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Small Test Dataset: Coefficient Comparison')
plt.legend()

plt.subplot(2, 2, 4)
plt.bar(range(len(custom_results_collinear.beta)), custom_results_collinear.beta, alpha=0.5, label='Custom Model')
plt.bar(range(len(sk_model_collinear.coef_)), sk_model_collinear.coef_, alpha=0.5, label='Sklearn Model')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Collinear Dataset: Coefficient Comparison')
plt.legend()
plt.tight_layout()


X_test = generate_data(n_samples=50)[0]
y_pred_homotopy = results.predict(X_test)
y_pred_sklearn = sk_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_pred_homotopy, label='LassoHomotopyModel Predictions')
plt.plot(y_pred_sklearn, label='Scikit-Learn Lasso Predictions')
plt.legend()
plt.title('Prediction Comparison')
plt.show()

