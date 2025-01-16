import numpy as np
import pandas as pd

# Create a dataframe using pandas from the dataset
df = pd.read_csv('DataSet/real_estate_dataset.csv')

# Get the number of samples and features from the csv
samples, features = df.shape

# Get the columns of the dataframe and store it in a text file
df_columns = df.columns
np.savetxt('DataSet/columns.txt', df_columns, fmt='%s')

# From the dataset use square_feet, Garage_size, Location_Score, Distance_to_center as feature for the model
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]

# Use price as the target variable
y = df['Price']

print(f"Shape of X: {X.shape}\n")
print(f"Data Type of X: {X.dtypes}\n")

samples,features = X.shape

# Build a linear model to predict price from the four features in X
# make an array of coefs of the size of features + 1, initialize to 1.
coefs = np.ones(features+1)

# Predict the price for each sample in X
predictions_bydefn = X @ coefs[1:] + coefs[0]

#Append a column of 1s to X
X = np.hstack((np.ones((samples,1)),X))

# Predict the price for each sample in X
predictions = X @ coefs

# Check if predictions and predictions_bydefn are the same
is_same = np.allclose(predictions, predictions_bydefn)

#print whether the predictions are the same in both cases
if is_same:
    print("The predictions are the same in both cases")

errors = y - predictions
#find relative error
rel_errors = errors / y

#Find mean squared error from error vector
squared_loss = errors.T @ errors
mean_squared_loss = squared_loss / samples

#printthe shape and norm of the error
print(f"Shape of errors: {errors.shape}")
print(f"Norm of errors: {np.linalg.norm(errors)}")
print(f"Norm of relative errors: {np.linalg.norm(rel_errors)}")

# What is the optimization problem that we are solving here?
# We are trying to minimize the mean squared loss by finding the optimal values of the coefficients
# These type of problems are called least square problems


# Aside
# IN heat transfer Nu = f(Re, Pr) : Nu = \alpha Re^m Pr^n, we want to find the values of m and n that minimize the error
# between the predicted Nu and the actual Nu. This is a least square problem
# Since it does not the linear regression criteria we use log on both size and then fit a equation on the data.

# Objective function: f(coefs) = 1/samples + \sum_{i=1}^{samples} (y_i - coefs^T x_i)^2

# What is a solution?
# The solution is the values of the coefficients that minimize the objective function
# In this case, the solution is the values of the coefficients that minimize the mean squared loss

# How do I find the solution?
# the solution have to satisfy the first order condition of the objective function i.e the gradient of the objective function must be zero at the solution

# get the loss matrix for the given data
loss_matrix = (y - X @ coefs).T @ (y - X @ coefs)/samples

# Calculate the gradient of the loss function
gradient = -2 * X.T @ (y - X @ coefs) / samples

# Setting the gradient to zero we get the follwoing equation
# X^T @ (X @ coefs - y) = 0
# X^T @ X @ coefs = X^T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# Save the coefficients to a text file
np.savetxt('DataSet/coefficients.txt', coefs, fmt='%s')

# Predict the price for each sample in X
optimal_solution_predictions = X @ coefs

# error model
optimal_solution_errors = y - optimal_solution_predictions

# Find relative error
optimal_solution_rel_errors = optimal_solution_errors / y

# print the norms of the errors
print(f"Norm of optimal solution errors: {np.linalg.norm(optimal_solution_errors)}")
print(f"Norm of optimal solution relative errors: {np.linalg.norm(optimal_solution_rel_errors)}")


# Use all the features in the dataset to build a linear model to predict price
y = df['Price'].values
X = df.drop('Price', axis=1).values
samples, features = X.shape

# Append a column of 1s to X
X = np.hstack((np.ones((samples,1)),X))

coefs = np.ones(features+1)
# Calculate the coefficients
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# Predict the price for each sample in X
predictions_with_all_features = X @ coefs

# Calculate the errors
errors_with_all_features = y - predictions_with_all_features

# Find the relative errors
rel_errors_with_all_features = errors_with_all_features / y

# Print the norms of the errors
print(f"Norm of errors with all features: {np.linalg.norm(errors_with_all_features)}")
print(f"Norm of relative errors with all features: {np.linalg.norm(rel_errors_with_all_features)}")

# Save the coefficients to a text file
np.savetxt('DataSet/coefficients_all_features.txt', coefs, fmt='%s')

# Solve the normal equation using Q R decomposition
Q,R = np.linalg.qr(X)

print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")

# Write R to a file named R.csv
np.savetxt('DataSet/R.csv', R, delimiter=',')

#Calculate Q.T @ Q and save it to a file named Q_TQ.csv
Q_TQ = Q.T @ Q
np.savetxt('DataSet/Q_TQ.csv', Q_TQ, delimiter=',')

# X = QR
# X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R @ coefs = Q.T @ y

b = Q.T @ y

# Solve for coefs using back substitution
# Use the loop to solve backsubstitution problem
coefs_backsub = np.zeros(features+1)
for i in range(features,-1,-1):
    coefs_backsub[i] = b[i]
    for j in range(i+1,features+1):
        coefs_backsub[i] = coefs_backsub[i] - R[i,j] * coefs_backsub[j]
    coefs_backsub[i] = coefs_backsub[i] / R[i,i]

#Save the coefficients to a text file
np.savetxt('DataSet/coefficients_qr_backsub.txt', coefs_backsub, fmt='%s')

#Solve the normal equation using the SVD decomposition
U, S, V_T = np.linalg.svd(X, full_matrices=False)

# Solving the normal equation using the SVD decomposition
# X = U S V_T
# X.T @ X = V S^2 V_T
# X.T @ y = V S U.T @ y
# coefs = V S^-1 U.T @ y

print(f"Shape of U: {U.shape}")
print(f"Shape of S: {S.shape}")
print(f"Shape of V_T: {V_T.shape}")


coeffs_svd = V_T.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

# Save the coefficients to a text file
np.savetxt('DataSet/coefficients_svd.txt', coeffs_svd, fmt='%s')

predictions_svd = X @ coeffs_svd

# Calculate the errors
errors_svd = y - predictions_svd

# Find the relative errors
rel_errors_svd = errors_svd / y

# Print the norms of the errors
print(f"Norm of errors with SVD: {np.linalg.norm(errors_svd)}")
print(f"Norm of relative errors with SVD: {np.linalg.norm(rel_errors_svd)}")


