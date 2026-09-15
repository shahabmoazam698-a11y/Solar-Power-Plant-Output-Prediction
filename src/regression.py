import numpy as np


# Prediction function
def predict(X, theta):
    return X @ theta


# Cost function
def cost_function(X, y, theta):

    error = y - predict(X, theta)

    J = 0.5 * np.sum(error ** 2)

    return J


# Normal equation
def normal_equation(X, y):

    theta = (
        np.linalg.inv(X.T @ X)
        @ X.T
        @ y
    )

    return theta


# Batch Gradient Descent
def batch_gradient_descent(
    X,
    y,
    alpha,
    iterations
):

    theta = np.zeros(X.shape[1])

    cost_history = []

    for i in range(iterations):

        prediction = predict(X, theta)

        error = y - prediction

        theta = theta + alpha * (
            X.T @ error
        )

        J = cost_function(X, y, theta)

        cost_history.append(J)

    return theta, cost_history


# Stochastic Gradient Descent
def stochastic_gradient_descent(
    X,
    y,
    alpha,
    epochs
):

    theta = np.zeros(X.shape[1])

    cost_history = []

    for epoch in range(epochs):

        for i in range(len(X)):

            xi = X[i]
            yi = y[i]

            prediction = xi @ theta

            error = yi - prediction

            theta = theta + (
                alpha * error * xi
            )

        J = cost_function(X, y, theta)

        cost_history.append(J)

    return theta, cost_history


# RMSE implemented by hand
def rmse(y, prediction):

    error = y - prediction

    return np.sqrt(
        np.mean(error ** 2)
    )