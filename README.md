# <b>Introduction</b>
This project implements LASSO (Least Absolute Shrinkage and Selection Operator) regression using the Homotopy Method . The implementation is done entirely from scratch, relying only on NumPy and SciPy for computations. No external libraries like scikit-learn were used.
The goal of this project is to solve the LASSO optimization problem using the Homotopy algorithm. This method is efficient and produces sparse solutions, making it ideal for feature selection in datasets with many features.

## Regularization:
Regularization helps in preventing overfitting by adding penalty term to loss function. Regularization has two types:
1. Ridge Regularization: It adds a penalty that is propotional to the square of magnitude of coefficient.
2. LASSO (Least Absolute Shrinkage and Selection Operator): It adds a penalty proportional to absolute value of coefficient.

Here, λ is a regularization parameter that controls the strength of penalty.
LASSO is helpful for feature selection. LASSO is handling high-dimensional data. It adds a penalty term to Residual sum of squares, then it is multiplied to lambda. The penalty helps in avoiding multi-collinearity and overfitting issues.
In both cases, λ (lambda) is the regularization parameter that controls the strength of the penalty:
1. A small λ results in minimal penalty, allowing the model to fit the training data closely (risking overfitting).
2. A large λ imposes a stronger penalty, shrinking coefficients toward zero and simplifying the model.

The Homotopy Method is an efficient algorithm for solving the LASSO problem. It works by:

1. Starting with a large value of λ , where all coefficients are zero.
2. Gradually decreasing λ while tracking the active set of features (features with non-zero coefficients).
3. Updating the coefficients of the active set at each step and checking for features entering or leaving the active set.

## Running the Code
To run the code in this project using VS Code , follow these steps:

### 1. Set Up a Virtual Environment
Open the terminal in VS Code by navigating to View > Terminal or using the shortcut Ctrl + ` (backtick).
Create a virtual environment using Python's venv module:
```
python -m venv venv
```
Activate the virtual environment:
```
venv\Scripts\activate
```
### 2. Install Required Packages
Once the virtual environment is activated, install the required dependencies from the requirements.txt file:
```
pip install -r requirements.txt
```
### 3. Run the Tests
Navigate to the LassoHomotopy/tests/ directory and execute the test suite using pytest.
First, navigate to the tests folder:
Run the tests using pytest. For example, to run the main test script:
```
pytest test_LassoHomotopy.py
```


# 1. What does the model you have implemented do and when should it be used?
This model performs LASSO regression using the Homotopy Method to minimize the residual sum of squares while adding a penalty on the absolute size of coefficients. This approach results in sparse solutions, setting some coefficients to zero, which helps with feature selection. It’s best suited for datasets with many features, especially when there’s multicollinearity or noise, and when interpretability is important. It works well in fields like genomics, finance, or text analysis, where identifying key predictors and avoiding overfitting are critical. Use it when you need a simpler, more interpretable model.

# 2. How did you test your model to determine if it is working reasonably correctly?
# 3. What parameters have you exposed to users of your implementation in order to tune performance?
The key parameters exposed in the implementation for tuning performance are:
alpha: 
This is the regularization parameter. It controls the strength of the L1 penalty applied to the model coefficients. A higher alpha enforces more regularization, often leading to sparser (i.e., more zeroed-out) coefficients, which can help reduce overfitting but may also underfit if set too high.

tol (tolerance):
This parameter sets the convergence threshold. The algorithm will stop iterating once the change in the model (or related objective function) is less than this value. Lowering the tolerance can lead to a more precise solution at the cost of additional iterations and computation time.

max_iter (maximum iterations):
This parameter defines the maximum number of iterations the algorithm will run. It serves as a safeguard to ensure the algorithm terminates even if the convergence criteria are not met, helping control computational time and resources.

These parameters allow users to balance between model accuracy and computational efficiency, adjusting the regularization strength, convergence precision, and iteration limits as needed for different datasets and performance requirements.
# 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
