# Contributors:
1. Madhusoodhan Tirunangur Girinarayanan (A20580122)
2. Mukund Sanjay Bharani (A20577945)
3. Muthu Nageswaran Kalyani Narayanamoorthy (A20588118)
4. Sabarish Raja Ramesh Raja (A20576363)
# <b>Introduction</b>
This project implements LASSO (Least Absolute Shrinkage and Selection Operator) regression using the Homotopy Method . The implementation is done entirely from scratch, relying only on NumPy and SciPy for computations. No external libraries like scikit-learn were used.
The goal of this project is to solve the LASSO optimization problem using the Homotopy algorithm. This method is efficient and produces sparse solutions, making it ideal for feature selection in datasets with many features.

## Regularization:
Regularization helps in preventing overfitting by adding penalty term to loss function. Regularization has two types:
1. Ridge Regularization: It adds a penalty that is propotional to the square of magnitude of coefficient.
2. LASSO (Least Absolute Shrinkage and Selection Operator): It adds a penalty proportional to absolute value of coefficient.

Here, λ is a regularization parameter that controls the strength of penalty.
## LASSO Regression
![image_alt](https://github.com/sabarishraja/Project1/blob/main/Lasso%20Equation.png?raw=true)

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
## Visualization output:
![image_alt](https://github.com/sabarishraja/Project1/blob/main/Visualization%20results.jpeg?raw=true)

![image_alt](https://github.com/sabarishraja/Project1/blob/main/Prediction%20comparison.jpeg?raw=true)

![image_alt](https://github.com/sabarishraja/Project1/blob/main/Coefficient%20comparison.jpeg?raw=true)
# 1. What does the model you have implemented do and when should it be used?
This model performs LASSO regression using the Homotopy Method to minimize the residual sum of squares while adding a penalty on the absolute size of coefficients. This approach results in sparse solutions, setting some coefficients to zero, which helps with feature selection. It’s best suited for datasets with many features, especially when there’s multicollinearity or noise, and when interpretability is important. This model is a great choice in the following situations:
* Picking Out Key Features :
When you’re dealing with datasets that have too many features (more features than data points), this model helps identify the most important ones, simplifying your analysis.
* Dealing with Correlated Variables :
If your data has features that are highly related to each other, this model can handle it by reducing redundancy and avoiding overfitting (e.g., as shown in collinear_data.csv).
* Making Sense of Results :
When you want a clear and interpretable model where only the most relevant features have non-zero coefficients, making it easy to understand which variables matter.
* Working with Smaller Datasets :
It’s well-suited for small to medium-sized datasets where computational speed isn’t a big concern. For very large datasets, more optimized tools like scikit-learn might be better.
# 2. How did you test your model to determine if it is working reasonably correctly?
To ensure the custom LassoHomotopyModel works as intended, the following steps were carried out in the code:
* The model’s predictions, coefficients, and performance metrics (MSE and R²) were compared with those from Scikit-Learn’s Lasso using datasets like small_test.csv and collinear_data.csv.
* Scatter plots of actual vs predicted values were created to visually confirm that both models produced similar results.
* Bar charts were used to compare the learned coefficients (custom_results_small.beta vs sklearn_model_small.coef_) to verify consistent feature selection.
* The model was tested on a variety of scenarios generated using the generate_data function, including collinear features, high-dimensional data, sparse data, noisy data, and datasets of different sizes.
* Different regularization strengths (alpha values of 0.01, 0.1, and 1.0) were evaluated to analyze their impact on the model’s behavior.
* Edge cases such as small datasets (n_samples=20) and non-negative features (X_non_negative = np.abs(X)) were validated to ensure robustness.
These steps confirmed that the custom model performs reliably and aligns with the behavior of Scikit-Learn’s implementation.
# 3. What parameters have you exposed to users of your implementation in order to tune performance?
The key parameters exposed in the implementation for tuning performance are:
* alpha: 
This is the regularization parameter. It controls the strength of the L1 penalty applied to the model coefficients. A higher alpha enforces more regularization, often leading to sparser (i.e., more zeroed-out) coefficients, which can help reduce overfitting but may also underfit if set too high.

* tol (tolerance):
This parameter sets the convergence threshold. The algorithm will stop iterating once the change in the model (or related objective function) is less than this value. Lowering the tolerance can lead to a more precise solution at the cost of additional iterations and computation time.

* max_iter (maximum iterations):
This parameter defines the maximum number of iterations the algorithm will run. It serves as a safeguard to ensure the algorithm terminates even if the convergence criteria are not met, helping control computational time and resources.

These parameters allow users to balance between model accuracy and computational efficiency, adjusting the regularization strength, convergence precision, and iteration limits as needed for different datasets and performance requirements.
# 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
When we check the graph we find that the custom lasso homotopy model clearly underperforms on "small_test.csv". the coefficient values diverge significantly as witnessed in the actual vs predicated plot using matplotlib. Also the result for collinearity is also slightly ambiguous as we only tested on only one csv file, "collinear_data.csv" even though it almost fits similar to scikit's model even though its a minor issue. Also for lasso penalty we have used L1 regularization and L2 penalty that is ridge regression partially, for stability but we could have used Elastic Net that combines both L1 and L2 regularization. Here we can also learn that the custom homotopy model dosent perform well in high noise of 1.0 as compared to sklearn model under same noise conditions which is also another setback
If provided more time we would first go for implementing elastic net in place of L1 and L2 regularization separately which would improve the performance giving more reliable performance of the model.  Then we go for testing the model for more colinear data to test the ambiguity of the model. We might solve the problem that we faced in small csv if switched to elastic net instead L1 regularization.
