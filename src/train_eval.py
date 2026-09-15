import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from regression import (
    predict,
    normal_equation,
    batch_gradient_descent,
    stochastic_gradient_descent,
    rmse
)


# ============================================================
# 1. LOAD DATA
# ============================================================

plant = pd.read_csv("data/plant1_hourly.csv")
weather = pd.read_csv("data/plant1_openmeteo.csv")

plant["datetime"] = pd.to_datetime(plant["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])

data = plant.merge(
    weather,
    on="datetime",
    how="inner"
)

print("Merged rows:", len(data))


# ============================================================
# 2. CREATE TIME FEATURES
# ============================================================

data["hour"] = data["datetime"].dt.hour

data["sin_hour"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["cos_hour"] = np.cos(
    2 * np.pi * data["hour"] / 24
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train = data[
    data["datetime"] <= "2020-06-10 23:59:59"
].copy()

test = data[
    data["datetime"] >= "2020-06-11"
].copy()

print("Training rows:", len(train))
print("Testing rows:", len(test))


# ============================================================
# 4. FEATURE SETS
# ============================================================

features_A = [
    "irradiation",
    "module_temp",
    "ambient_temp",
    "sin_hour",
    "cos_hour"
]

features_B = [
    "sw_radiation",
    "temp_2m",
    "cloud_cover",
    "sin_hour",
    "cos_hour"
]

target = "ac_power"


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

def prepare_features(train, test, feature_names):

    X_train = train[
        feature_names
    ].to_numpy(dtype=float)

    X_test = test[
        feature_names
    ].to_numpy(dtype=float)

    # Scaling values calculated only from training data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    # Standardization
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Add intercept x0 = 1
    X_train = np.column_stack(
        (
            np.ones(len(X_train)),
            X_train
        )
    )

    X_test = np.column_stack(
        (
            np.ones(len(X_test)),
            X_test
        )
    )

    return X_train, X_test, mean, std


# ============================================================
# 6. PREPARE SET A AND SET B
# ============================================================

X_train_A, X_test_A, mean_A, std_A = prepare_features(
    train,
    test,
    features_A
)

X_train_B, X_test_B, mean_B, std_B = prepare_features(
    train,
    test,
    features_B
)

y_train = train[
    target
].to_numpy(dtype=float)

y_test = test[
    target
].to_numpy(dtype=float)


# ============================================================
# 7. NORMAL EQUATION
# ============================================================

theta_normal_A = normal_equation(
    X_train_A,
    y_train
)

theta_normal_B = normal_equation(
    X_train_B,
    y_train
)

print("\nNormal Equation Set A theta:")
print(theta_normal_A)

print("\nNormal Equation Set B theta:")
print(theta_normal_B)


# ============================================================
# 8. BATCH GD LEARNING RATE TEST - SET A
# ============================================================

# Assignment learning-rate experiment:
# keep this at 500 iterations.

batch_alphas = [
    1e-5,
    1e-4,
    1e-3
]

batch_histories = {}

for alpha in batch_alphas:

    theta_temp, history = batch_gradient_descent(
        X_train_A,
        y_train,
        alpha,
        500
    )

    batch_histories[alpha] = history

    print(
        "\nBatch alpha:",
        alpha,
        "Final cost:",
        history[-1]
    )


# ============================================================
# 9. BATCH LEARNING-RATE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

for alpha in batch_alphas:

    plt.plot(
        batch_histories[alpha],
        label=str(alpha)
    )

plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Batch Gradient Descent Learning Rates")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/batch_learning_rates.png"
)

plt.close()


# ============================================================
# 10. FINAL BATCH GD
# ============================================================

# 1e-4 was selected from the learning-rate experiment.
batch_alpha = 1e-4

# Longer final run to check convergence with Normal Equation.
batch_final_iterations = 50000

theta_batch_A, batch_cost_A = batch_gradient_descent(
    X_train_A,
    y_train,
    batch_alpha,
    batch_final_iterations
)

theta_batch_B, batch_cost_B = batch_gradient_descent(
    X_train_B,
    y_train,
    batch_alpha,
    batch_final_iterations
)


# ============================================================
# 11. COMPARE NORMAL AND BATCH THETA
# ============================================================

difference_A = np.max(
    np.abs(
        theta_normal_A - theta_batch_A
    )
)

difference_B = np.max(
    np.abs(
        theta_normal_B - theta_batch_B
    )
)

print(
    "\nMax theta difference Set A:",
    difference_A
)

print(
    "Max theta difference Set B:",
    difference_B
)


# ============================================================
# 12. SGD LEARNING RATE TEST - SET A
# ============================================================

sgd_alphas = [
    1e-4,
    1e-3,
    1e-2
]

sgd_histories = {}

for alpha in sgd_alphas:

    theta_temp, history = stochastic_gradient_descent(
        X_train_A,
        y_train,
        alpha,
        50
    )

    sgd_histories[alpha] = history

    print(
        "\nSGD alpha:",
        alpha,
        "Final cost:",
        history[-1]
    )


# ============================================================
# 13. SGD LEARNING-RATE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

for alpha in sgd_alphas:

    plt.plot(
        sgd_histories[alpha],
        label=str(alpha)
    )

plt.xlabel("Epoch")
plt.ylabel("Cost J")
plt.title("SGD Learning Rates")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/sgd_learning_rates.png"
)

plt.close()


# ============================================================
# 14. FINAL SGD
# ============================================================

# Keep the current selected value for this verification run.
sgd_alpha = 1e-3

theta_sgd_A, sgd_cost_A = stochastic_gradient_descent(
    X_train_A,
    y_train,
    sgd_alpha,
    50
)

theta_sgd_B, sgd_cost_B = stochastic_gradient_descent(
    X_train_B,
    y_train,
    sgd_alpha,
    50
)

print("\n========== FINAL THETA VALUES ==========")

print("\nSet A Normal theta:")
print(theta_normal_A)

print("\nSet A Batch theta:")
print(theta_batch_A)

print("\nSet A SGD theta:")
print(theta_sgd_A)

# ============================================================
# 15. TEST PREDICTIONS
# ============================================================

pred_normal_A = predict(
    X_test_A,
    theta_normal_A
)

pred_batch_A = predict(
    X_test_A,
    theta_batch_A
)

pred_sgd_A = predict(
    X_test_A,
    theta_sgd_A
)

pred_normal_B = predict(
    X_test_B,
    theta_normal_B
)

pred_batch_B = predict(
    X_test_B,
    theta_batch_B
)

pred_sgd_B = predict(
    X_test_B,
    theta_sgd_B
)


# ============================================================
# 16. CLIP NEGATIVE PREDICTIONS
# ============================================================

pred_normal_A = np.maximum(pred_normal_A, 0)
pred_batch_A = np.maximum(pred_batch_A, 0)
pred_sgd_A = np.maximum(pred_sgd_A, 0)

pred_normal_B = np.maximum(pred_normal_B, 0)
pred_batch_B = np.maximum(pred_batch_B, 0)
pred_sgd_B = np.maximum(pred_sgd_B, 0)


# ============================================================
# 17. ALL-HOUR RMSE
# ============================================================

print("\n========== ALL-HOUR RMSE ==========")

print(
    "Set A Normal:",
    rmse(y_test, pred_normal_A)
)

print(
    "Set A Batch:",
    rmse(y_test, pred_batch_A)
)

print(
    "Set A SGD:",
    rmse(y_test, pred_sgd_A)
)

print(
    "Set B Normal:",
    rmse(y_test, pred_normal_B)
)

print(
    "Set B Batch:",
    rmse(y_test, pred_batch_B)
)

print(
    "Set B SGD:",
    rmse(y_test, pred_sgd_B)
)


# ============================================================
# 18. DAYTIME RMSE
# ============================================================

day_mask = (
    test["irradiation"].to_numpy() > 0
)

y_day = y_test[
    day_mask
]

print("\n========== DAYTIME RMSE ==========")

print(
    "Set A Normal:",
    rmse(
        y_day,
        pred_normal_A[day_mask]
    )
)

print(
    "Set A Batch:",
    rmse(
        y_day,
        pred_batch_A[day_mask]
    )
)

print(
    "Set A SGD:",
    rmse(
        y_day,
        pred_sgd_A[day_mask]
    )
)

print(
    "Set B Normal:",
    rmse(
        y_day,
        pred_normal_B[day_mask]
    )
)

print(
    "Set B Batch:",
    rmse(
        y_day,
        pred_batch_B[day_mask]
    )
)

print(
    "Set B SGD:",
    rmse(
        y_day,
        pred_sgd_B[day_mask]
    )
)


# ============================================================
# 19. ACTUAL VS PREDICTED - SET A
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    test["datetime"],
    y_test,
    label="Actual"
)

plt.plot(
    test["datetime"],
    pred_normal_A,
    label="Predicted"
)

plt.xlabel("Date")
plt.ylabel("AC Power")

plt.title(
    "Actual vs Predicted - Set A Normal Equation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/actual_predicted_setA.png"
)

plt.close()


# ============================================================
# 20. ACTUAL VS PREDICTED - SET B
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    test["datetime"],
    y_test,
    label="Actual"
)

plt.plot(
    test["datetime"],
    pred_normal_B,
    label="Predicted"
)

plt.xlabel("Date")
plt.ylabel("AC Power")

plt.title(
    "Actual vs Predicted - Set B Normal Equation"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/actual_predicted_setB.png"
)

plt.close()


# ============================================================
# 21. RESIDUALS VS HOUR - SET A NORMAL
# ============================================================

residuals = (
    y_test - pred_normal_A
)

test_hours = (
    test["datetime"].dt.hour.to_numpy()
)

plt.figure(figsize=(8, 5))

plt.scatter(
    test_hours,
    residuals,
    alpha=0.5
)

plt.xlabel("Hour of Day")
plt.ylabel("Residual (Actual - Predicted)")

plt.title(
    "Residuals vs Hour - Set A"
)

plt.xticks(range(0, 24))

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/residuals_vs_hour.png"
)

plt.close()


# ============================================================
# 22. SAVE MODEL INFORMATION FOR TASK 6
# ============================================================

np.save(
    "results/setB_theta.npy",
    theta_normal_B
)

np.save(
    "results/setB_mean.npy",
    mean_B
)

np.save(
    "results/setB_std.npy",
    std_B
)


# ============================================================
# 23. FINISH
# ============================================================

print("\nTask 4 finished.")