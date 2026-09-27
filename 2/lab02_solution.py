# ============================================
# CSE-4155 Introduction to Machine Learning
# Lab 02 - Multiple Linear Regression, Cross Validation, and Polynomial Regression
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CCPP_PATH = ROOT / "CCPP" / "CCPP" / "Folds5x2_pp.xlsx"
DATA_02B_PATH = ROOT / "data_02b.csv"


# ============================================
# 1. LOAD DATA
# ============================================

def load_ccpp_data(path=CCPP_PATH):
    """
    Load the combined cycle power plant dataset.
    """
    df = pd.read_excel(path)
    return df


def save_single_feature_data(df, path=DATA_02B_PATH, feature_col="AT", target_col="PE"):
    """
    Create a single-feature dataset for the polynomial regression part.
    The original workspace does not contain data_02b.csv, so we create it
    from the supplied CCPP dataset using AT as the feature and PE as target.
    """
    subset = df[[feature_col, target_col]].copy()
    subset.columns = ["x", "y"]
    subset.to_csv(path, index=False)
    print(f"Saved single-feature dataset to {path}")


# ============================================
# 2. DATA PREPARATION
# ============================================

def add_bias(X):
    """
    Add bias term x0 = 1.
    """
    return np.column_stack((np.ones(len(X)), X))


def standardize_features(X_train, X_test=None):
    """
    Standardize features using training-set statistics.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std

    if X_test is None:
        return X_train_scaled, mean, std

    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def split_train_validation(X, y, val_fraction=0.2, seed=42):
    """
    Randomly split data into training and validation sets.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    val_size = int(len(X) * val_fraction)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ============================================
# 3. LINEAR REGRESSION TOOLS
# ============================================

def compute_cost(X, y, theta):
    """
    Compute cost J(theta) = 1/(2m) sum((X theta - y)^2).
    """
    m = len(y)
    errors = X.dot(theta) - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


def fit_linear_regression(X_train, y_train, X_val, y_val, learning_rate=0.01, iterations=2000):
    """
    Train linear regression with gradient descent and monitor train/validation curves.
    Return the best theta according to validation error.
    """
    m = len(y_train)
    theta = np.zeros(X_train.shape[1])

    train_history = []
    validation_history = []

    best_theta = theta.copy()
    best_train_cost = np.inf
    best_val_cost = np.inf

    for i in range(iterations):
        predictions = X_train.dot(theta)
        errors = predictions - y_train
        gradient = (1 / m) * X_train.T.dot(errors)
        theta = theta - learning_rate * gradient

        train_cost = compute_cost(X_train, y_train, theta)
        val_cost = compute_cost(X_val, y_val, theta)

        train_history.append(train_cost)
        validation_history.append(val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            best_train_cost = train_cost
            best_theta = theta.copy()

    return best_theta, train_history, validation_history, best_train_cost, best_val_cost


def print_model_summary(theta, feature_names=None, scale_params=None):
    """
    Print the learned parameters in a format similar to the Lab 1 notebook.
    """
    print("=" * 65)
    print("LEARNED MODEL PARAMETERS")
    print("=" * 65)

    if feature_names is None:
        feature_names = [f"theta_{i}" for i in range(len(theta))]

    if scale_params is not None:
        mean = scale_params["mean"]
        std = scale_params["std"]
        original_theta = theta.copy()
        original_theta[1:] = theta[1:] / std
        original_theta[0] = theta[0] - np.sum((theta[1:] * mean) / std)

        print("Original-scale parameters:")
        print(f"theta_0 = {original_theta[0]:.6f}")
        for i, name in enumerate(feature_names, start=1):
            print(f"{name} = {original_theta[i]:.6f}")

        print("\nEquation:")
        eq = f"y_hat = {original_theta[0]:.6f}"
        for i, name in enumerate(feature_names, start=1):
            eq += f" + ({original_theta[i]:.6f})*{name}"
        print(eq)
    else:
        print("Scaled/working parameters:")
        for i, val in enumerate(theta):
            if i == 0:
                print(f"theta_0 = {val:.6f}")
            else:
                print(f"theta_{i} = {val:.6f}")

    print("=" * 65)


# ============================================
# 4. PLOTTING
# ============================================

def plot_feature_vs_target(df, target_col="PE"):
    """
    Plot each feature against the target variable.
    """
    feature_cols = ["AT", "V", "AP", "RH"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, feature in zip(axes, feature_cols):
        ax.scatter(df[feature], df[target_col], s=10, alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
        ax.set_title(f"{feature} vs {target_col}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_error_curves(train_history, validation_history, title="Training and Validation Error Curves"):
    """
    Plot cost vs iteration for training and validation sets.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Training Error", color="blue")
    plt.plot(validation_history, label="Validation Error", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_polynomial_fit(x, y, x_fit, y_fit, degree, title):
    """
    Plot the fitted polynomial curve against the actual data.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=20, alpha=0.7, label="Data")
    plt.plot(x_fit, y_fit, color="red", linewidth=2, label=f"Degree {degree} fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_polynomial_errors(errors_by_degree):
    """
    Bar plot of validation errors for d = 1, 2, 3.
    """
    degrees = list(errors_by_degree.keys())
    values = [errors_by_degree[d] for d in degrees]

    plt.figure(figsize=(8, 5))
    plt.bar(degrees, values, color=["steelblue", "darkorange", "forestgreen"])
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Validation Error")
    plt.title("Validation Error by Polynomial Degree")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================
# 5. A) MULTIPLE LINEAR REGRESSION
# ============================================

def linear_regression_experiment(df):
    """
    Run the main multiple-variable linear regression task.
    """
    feature_cols = ["AT", "V", "AP", "RH"]
    X = df[feature_cols].values.astype(float)
    y = df["PE"].values.astype(float)

    X_train, y_train, X_val, y_val = split_train_validation(X, y, val_fraction=0.2, seed=42)

    # Raw feature experiment
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    theta_raw, train_hist_raw, val_hist_raw, train_cost_raw, val_cost_raw = fit_linear_regression(
        add_bias(X_train_raw),
        y_train,
        add_bias(X_val_raw),
        y_val,
        learning_rate=0.01,
        iterations=2000
    )

    # Standardized feature experiment
    X_train_scaled, X_val_scaled, mean, std = standardize_features(X_train, X_val)
    X_train_scaled = add_bias(X_train_scaled)
    X_val_scaled = add_bias(X_val_scaled)
    theta_scaled, train_hist_scaled, val_hist_scaled, train_cost_scaled, val_cost_scaled = fit_linear_regression(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        learning_rate=0.01,
        iterations=2000
    )

    print("=" * 65)
    print("PART A: MULTIPLE LINEAR REGRESSION")
    print("=" * 65)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("\nWithout feature scaling:")
    print(f"Best validation error: {val_cost_raw:.6f}")
    print(f"Training error at best validation point: {train_cost_raw:.6f}")
    print("Parameters:")
    for i, val in enumerate(theta_raw):
        print(f"theta_{i} = {val:.6f}")

    print("\nWith feature scaling:")
    print(f"Best validation error: {val_cost_scaled:.6f}")
    print(f"Training error at best validation point: {train_cost_scaled:.6f}")
    print("Parameters:")
    for i, val in enumerate(theta_scaled):
        print(f"theta_{i} = {val:.6f}")

    print("\nObservation: Standardized features generally converge more smoothly and often yield a lower and more stable validation error.")

    plot_error_curves(train_hist_raw, val_hist_raw, title="Linear Regression: Training and Validation Error Curves (Raw Features)")
    plot_error_curves(train_hist_scaled, val_hist_scaled, title="Linear Regression: Training and Validation Error Curves (Scaled Features)")

    return {
        "theta_raw": theta_raw,
        "theta_scaled": theta_scaled,
        "raw_cost": (train_cost_raw, val_cost_raw),
        "scaled_cost": (train_cost_scaled, val_cost_scaled),
        "feature_means": mean,
        "feature_stds": std,
    }


# ============================================
# 6. B) K-FOLD CROSS VALIDATION
# ============================================

def k_fold_cross_validation(X, y, k=5, learning_rate=0.01, iterations=2000, seed=42):
    """
    Implement 5-fold cross-validation for multiple linear regression.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    folds = np.array_split(indices, k)

    fold_errors = []

    for fold_idx in range(k):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        X_train_scaled, X_val_scaled, _, _ = standardize_features(X_train, X_val)
        X_train_scaled = add_bias(X_train_scaled)
        X_val_scaled = add_bias(X_val_scaled)

        theta, _, _, _, val_cost = fit_linear_regression(X_train_scaled, y_train, X_val_scaled, y_val, learning_rate, iterations)
        fold_errors.append(val_cost)

    mean_val_error = float(np.mean(fold_errors))
    std_val_error = float(np.std(fold_errors))

    print("=" * 65)
    print("PART B: K-FOLD CROSS VALIDATION")
    print("=" * 65)
    print(f"5-fold CV mean validation error: {mean_val_error:.6f}")
    print(f"5-fold CV std. dev.: {std_val_error:.6f}")
    print(f"Per-fold validation errors: {np.round(fold_errors, 6)}")

    return mean_val_error, std_val_error, fold_errors


# ============================================
# 7. C) POLYNOMIAL REGRESSION
# ============================================

def polynomial_design_matrix(x, degree):
    """
    Create a polynomial feature matrix for one variable.
    """
    x = np.asarray(x, dtype=float)
    X = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        X = np.column_stack((X, x ** d))
    return X


def polynomial_regression_experiment(df):
    """
    Tune polynomial degree d = 1, 2, 3 and compare validation errors.
    """
    x = df["x"].values.astype(float)
    y = df["y"].values.astype(float)

    X_train, y_train, X_val, y_val = split_train_validation(x.reshape(-1, 1), y, val_fraction=0.2, seed=42)

    result = {}
    best_degree = None
    best_val_error = np.inf

    for degree in [1, 2, 3]:
        X_train_poly = polynomial_design_matrix(X_train.ravel(), degree)
        X_val_poly = polynomial_design_matrix(X_val.ravel(), degree)

        theta, train_history, validation_history, train_cost, val_cost = fit_linear_regression(
            X_train_poly,
            y_train,
            X_val_poly,
            y_val,
            learning_rate=0.01,
            iterations=2000
        )

        result[degree] = {
            "theta": theta,
            "train_history": train_history,
            "validation_history": validation_history,
            "train_cost": train_cost,
            "val_cost": val_cost,
        }

        if val_cost < best_val_error:
            best_val_error = val_cost
            best_degree = degree

        print("=" * 65)
        print(f"PART C: DEGREE = {degree}")
        print("=" * 65)
        print(f"Best validation error: {val_cost:.6f}")
        print(f"Training error: {train_cost:.6f}")
        print("Parameters:")
        for i, val in enumerate(theta):
            print(f"theta_{i} = {val:.6f}")

        x_fit = np.linspace(x.min(), x.max(), 300)
        X_fit = polynomial_design_matrix(x_fit, degree)
        y_fit = X_fit.dot(theta)
        plot_polynomial_fit(x, y, x_fit, y_fit, degree, title=f"Polynomial Regression (Degree {degree})")
        plot_error_curves(train_history, validation_history, title=f"Degree {degree} Error Curves")

    print("=" * 65)
    print("PART C: BEST POLYNOMIAL DEGREE")
    print("=" * 65)
    print(f"Best polynomial degree based on validation error: d = {best_degree}")
    print(f"Best validation error: {best_val_error:.6f}")

    val_errors = {d: result[d]["val_cost"] for d in [1, 2, 3]}
    plot_polynomial_errors(val_errors)

    return result


# ============================================
# 8. MAIN
# ============================================

def main():
    """
    Execute the full Lab 02 workflow.
    """
    print("Loading the CCPP dataset...")
    df = load_ccpp_data()
    print(df.head())
    print("\nDataset shape:", df.shape)

    plot_feature_vs_target(df)

    # Run Part A
    linear_regression_experiment(df)

    # Run Part B
    X = df[["AT", "V", "AP", "RH"]].values.astype(float)
    y = df["PE"].values.astype(float)
    k_fold_cross_validation(X, y, k=5, learning_rate=0.01, iterations=2000, seed=42)

    # Generate the single-feature file if needed
    save_single_feature_data(df, feature_col="AT", target_col="PE")

    # Run Part C
    d02_df = pd.read_csv(DATA_02B_PATH)
    polynomial_regression_experiment(d02_df)

    print("\nLab 02 workflow completed successfully.")


if __name__ == "__main__":
    main()
