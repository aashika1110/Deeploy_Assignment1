import numpy as np


def ridge_regression(X, y, alpha):
    """
    Perform Ridge Regression using the closed-form solution.

    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target vector (n_samples)
    - alpha: Regularization strength (lambda)

    Returns:
    - Coefficients (weights) of the regression model.
    """
    # Add bias term (intercept)
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for the intercept

    # Closed-form solution: beta = (X.T @ X + alpha * I)^(-1) @ X.T @ y
    n_features = X.shape[1]
    I = np.eye(n_features)  # Identity matrix
    I[0, 0] = 0  # Don't regularize the intercept term
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

    return beta


# Input the number of features and samples
n_features = int(input("Enter the number of features: "))
n_samples = int(input("Enter the number of data points (samples): "))

# Input the feature matrix
print(f"Enter the feature matrix ({n_samples} rows, {n_features} columns):")
X = np.array([list(map(float, input().split())) for _ in range(n_samples)])

# Input the target vector
print(f"Enter the target vector ({n_samples} rows):")
y = np.array([float(input()) for _ in range(n_samples)])

# Input the regularization strength
alpha = float(input("Enter the regularization strength (lambda): "))

# Perform Ridge Regression
coefficients = ridge_regression(X, y, alpha)

# Display the coefficients
print("\nThe coefficients (including intercept) are:")
print(coefficients)

# Predict using the coefficients
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept to input
predictions = X_with_intercept @ coefficients

print("\nPredicted values for the given input data:")
print(predictions)
