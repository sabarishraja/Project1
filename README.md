LASSO Regression with Homotopy Method
Introduction

LASSO regression extends ordinary least squares (OLS) by adding an L1 penalty to the loss function:

Loss= 
2n
1
​
 ∥y−Xθ∥ 
2
2
​
 +α∥θ∥ 
1
​
 
Where:

y: Target values
X: Feature matrix
θ: Model coefficients
α: Regularization strength (controls sparsity)
The L1 penalty encourages sparsity by driving some coefficients to exactly zero, effectively performing feature selection.

Why Use the Homotopy Method?
The Homotopy Method solves the LASSO problem by:

Starting with an empty active set of features.
Iteratively adding/removing features based on their correlation with the residual.
Solving a least-squares problem on the active set and applying soft thresholding.
This approach ensures computational efficiency and adaptability to sequential data updates.

Setup Instructions
Prerequisites
Python 3.8 or higher
Virtual environment tool (venv or conda)
Dependencies listed in requirements.txt

Steps
1. Clone the repository:
git clone https://github.com/your-username/LassoHomotopy.git
cd LassoHomotopy

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

3. Install dependencies:
pip install -r requirements.txt

Contributors:
1. Madhusoodhan
2. Mukundh
3. Muthu
4. Sabarish Raja

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
