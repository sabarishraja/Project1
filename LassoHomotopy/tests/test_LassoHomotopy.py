import csv
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

# Load data from CSV
data = []
with open("tests/small_test.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
y = np.array([float(datum['y']) for datum in data])

# Fit the model
model = LassoHomotopyModel(alpha=0.1)
results = model.fit(X, y)

# Predict and validate
preds = results.predict(X)
print("Predictions:", preds)