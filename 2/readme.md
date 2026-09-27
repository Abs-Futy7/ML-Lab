# CSE-4155 Introduction to Machine Learning Lab 02
## Detailed Explanation, Code Walkthrough, Results Interpretation, and Viva Q&A

**Lab:** Lab 02 — Multiple Linear Regression, K-Fold Cross Validation, and Polynomial Regression  
**Course:** CSE-4155 Introduction to Machine Learning Lab  
**Purpose of this note:** Understand the Lab 02 implementation deeply enough to explain the code, mathematics, experimental results, and design decisions during lab evaluation/viva.

---

# 1. First understand the entire Lab 02 in one picture

Lab 02 extends the ideas from Lab 01.

In Lab 01, the model had only one input feature:

\[
\hat y = \theta_0 + \theta_1x
\]

Lab 02 adds three major ideas:

1. **Multiple Linear Regression** — several input variables instead of one.
2. **K-Fold Cross Validation** — evaluate the model using several train/validation partitions.
3. **Polynomial Regression** — represent a nonlinear relationship using polynomial features.

The full workflow is:

```text
                           LAB 02
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   Multiple Linear       5-Fold Cross       Polynomial
     Regression           Validation         Regression
          |                   |                   |
          v                   v                   v
   AT, V, AP, RH        Split data into       x -> x, x², x³
          |               five folds                |
          v                   |                     v
       Predict PE             v               Try d=1,2,3
          |              Train 5 times              |
          v                   |                     v
 Train/Validation Split       v              Validation Error
          |              Validation errors          |
          v              + parameters               v
 Feature Scaling              |                Choose best d
          |                   v                     |
          v               Average result             v
   Gradient Descent                              Best model
          |
          v
 Training + Validation Error
          |
          v
 Learned Parameters
```

---

# 2. What exactly does the lab require?

The lab has three parts.

## Part A — Multiple Linear Regression

The power-plant dataset contains four input features:

- `AT` — Ambient Temperature
- `V` — Exhaust Vacuum
- `AP` — Ambient Pressure
- `RH` — Relative Humidity

Target:

- `PE` — Net electrical energy output

The tasks are:

1. Plot each feature against the target.
2. Train a multiple linear regression model.
3. Plot training and validation error curves.
4. Report the best validation error.
5. Report training error.
6. Report learned parameters.
7. Compare results before and after feature scaling.

---

## Part B — 5-Fold Cross Validation

Use the same multiple-regression problem, but instead of relying on one train/validation split:

- divide the data into 5 folds,
- train 5 times,
- use a different fold as validation each time,
- compare learned parameters with the previous single-split experiment.

---

## Part C — Polynomial Regression

Use `data_02b.csv`.

There is one feature \(x\) and one target \(y\).

Try:

\[
d=1,\quad d=2,\quad d=3
\]

Then:

- choose the best degree using validation error,
- plot all three fitted curves on one graph,
- create a bar chart of the three validation errors,
- report the learned parameters of the best model,
- plot training and validation error curves for all three degrees.

---

# 3. Dataset used in Part A

The power-plant data has:

```text
Number of samples = 9568
Number of features = 4
```

The columns are:

```text
AT   V   AP   RH   PE
```

Example:

```text
AT      V       AP       RH      PE
14.96   41.76   1024.07  73.17   463.26
25.18   62.96   1020.04  59.08   444.37
 5.11   39.40   1012.16  92.14   488.56
```

The basic idea is:

```text
AT ─┐
V  ─┤
AP ─┼──> Linear Regression Model ───> PE
RH ─┘
```

---

# 4. Multiple linear regression

With one variable, Lab 01 used:

\[
\hat y=\theta_0+\theta_1x
\]

With four variables, Lab 02 uses:

\[
\boxed{
\hat y=
\theta_0+
\theta_1 AT+
\theta_2 V+
\theta_3 AP+
\theta_4 RH
}
\]

Here:

- \(\theta_0\) = intercept/bias
- \(\theta_1\) = coefficient for AT
- \(\theta_2\) = coefficient for V
- \(\theta_3\) = coefficient for AP
- \(\theta_4\) = coefficient for RH

---

# 5. Matrix form of multiple linear regression

For one example:

\[
\hat y=
\theta_0+
\theta_1x_1+
\theta_2x_2+
\theta_3x_3+
\theta_4x_4
\]

Using a dummy feature \(x_0=1\):

\[
X=
\begin{bmatrix}
1 & AT_1 & V_1 & AP_1 & RH_1\\
1 & AT_2 & V_2 & AP_2 & RH_2\\
\vdots & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}
\]

and

\[
\theta=
\begin{bmatrix}
\theta_0\\
\theta_1\\
\theta_2\\
\theta_3\\
\theta_4
\end{bmatrix}
\]

Therefore:

\[
\boxed{\hat y=X\theta}
\]

This is the same idea as Lab 01, only now `X` has five columns after adding the dummy feature.

---

# 6. Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
```

## NumPy

Used for:

- arrays,
- matrix multiplication,
- means and standard deviations,
- random permutations,
- polynomial powers,
- vectorized operations.

## Pandas

Used for:

- reading the Excel dataset,
- reading CSV data,
- displaying tables,
- presenting cross-validation results.

## Matplotlib

Used for:

- feature-vs-target scatter plots,
- training/validation curves,
- polynomial curves,
- validation-error bar chart.

## Why seed 42?

To make random splits reproducible.

The value `42` itself has no mathematical importance.

---

# 7. `load_power_plant_data()`

```python
def load_power_plant_data(filename, sheet_name="Sheet1"):
    data = pd.read_excel(filename, sheet_name=sheet_name)

    feature_names = ["AT", "V", "AP", "RH"]
    target_name = "PE"

    X = data[feature_names].values.astype(float)
    y = data[target_name].values.astype(float)

    return X, y, feature_names, target_name, data
```

This function separates the table into:

```text
X = input features
y = target
```

So:

```text
X.shape = (9568, 4)
y.shape = (9568,)
```

## Why use `.values`?

It converts a Pandas column/DataFrame into a NumPy array.

## Why `.astype(float)`?

Gradient descent involves decimal arithmetic, so numeric values are explicitly converted to floating point.

---

# 8. Plotting each feature against PE

The lab asks for four plots:

```text
AT vs PE
V  vs PE
AP vs PE
RH vs PE
```

The purpose is not training.

The purpose is **exploratory visualization**.

We want to visually inspect relationships between each individual feature and the target.

For example:

```python
plt.scatter(X[:, j], y)
```

`X[:, j]` means:

```text
all rows,
column j
```

So if `j = 0`, we take all `AT` values.

---

# 9. Why use scatter plots?

Because each row is one observation.

A scatter plot helps us see:

- positive relationships,
- negative relationships,
- nonlinear patterns,
- spread/noise,
- possible outliers.

It does not prove causation.

---

# 10. Train-validation split

The function:

```python
def train_validation_split(X, y, validation_ratio=0.20, seed=42):
```

creates:

```text
80% training data
20% validation data
```

Actual result:

```text
Training samples   = 7655
Validation samples = 1913
```

---

# 11. Why do we need a validation set?

The **training set** is used to update model parameters.

The **validation set** is not used for gradient updates.

Instead, it is used to check how well the current model performs on unseen data.

That gives us a better idea of generalization.

---

# 12. Why shuffle before splitting?

If the dataset is ordered in some way and we simply take the first 80% for training and last 20% for validation, the two sets may have different distributions.

Therefore we do:

```python
indices = rng.permutation(len(y))
```

This randomly permutes sample indices before splitting.

---

# 13. Data leakage — VERY important

The validation set must behave like unseen data.

Therefore:

> We must not calculate feature-scaling mean and standard deviation from the validation set.

In the notebook:

```python
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
```

Only the training data is used.

Then:

```python
X_train_scaled = (X_train - mean) / std
X_val_scaled   = (X_val - mean) / std
```

The validation data uses the **training mean and training standard deviation**.

That is correct.

---

# 14. What would be wrong with scaling before splitting?

Suppose we compute:

```python
mean = np.mean(X_all)
```

before the train-validation split.

Then information from validation samples influences the transformation used during training.

That is called **data leakage**.

Even though the leakage may appear small, it violates the idea that validation data should remain unseen.

---

# 15. `process_multivariable_data()`

This function has two main responsibilities:

```text
1. Optional feature standardization
2. Add dummy feature x0 = 1
```

---

# 16. Feature standardization

For every feature \(j\):

\[
\boxed{
z_j=\frac{x_j-\mu_j}{\sigma_j}
}
\]

where:

- \(\mu_j\) = mean of feature \(j\) in training data
- \(\sigma_j\) = standard deviation of feature \(j\) in training data

Because there are four features, we calculate four means and four standard deviations.

---

# 17. Why use `axis=0`?

```python
mean = np.mean(X_train, axis=0)
```

`X_train` looks conceptually like:

```text
AT    V    AP    RH
AT    V    AP    RH
AT    V    AP    RH
...
```

`axis=0` means:

> Go down the rows and compute one mean for each column.

Result:

```text
mean = [mean_AT, mean_V, mean_AP, mean_RH]
```

The same applies to standard deviation.

---

# 18. Why do we check `std == 0`?

```python
std[std == 0] = 1.0
```

If a feature has exactly the same value for every training example:

\[
\sigma=0
\]

Then:

\[
\frac{x-\mu}{0}
\]

would cause division by zero.

Setting its standard deviation to `1.0` prevents numerical failure.

---

# 19. Dummy feature

After scaling, the notebook adds:

```python
np.ones(len(X_train_processed))
```

So each row becomes:

```text
[1, AT, V, AP, RH]
```

or, if scaled:

```text
[1, z_AT, z_V, z_AP, z_RH]
```

The first column handles the intercept.

---

# 20. Why don't we standardize the dummy feature?

Because it is not a physical feature.

It is intentionally fixed to:

```text
1, 1, 1, 1, ...
```

so that:

\[
1\times\theta_0=\theta_0
\]

remains the intercept.

---

# 21. Cost function

The notebook keeps the same cost function used in Lab 01:

\[
\boxed{
J(\theta)
=
\frac{1}{2m}
\sum_{i=1}^{m}
(\hat y_i-y_i)^2
}
\]

Code:

```python
def compute_cost(X, y, theta):
    m = len(y)

    predictions = X.dot(theta)
    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost
```

---

# 22. What does `X.dot(theta)` do?

It computes predictions for all examples at once:

\[
\boxed{\hat y=X\theta}
\]

This is vectorization.

No Python loop is required for individual samples.

---

# 23. Error

```python
errors = predictions - y
```

For sample \(i\):

\[
e_i=\hat y_i-y_i
\]

If:

```text
prediction = 450
actual     = 455
```

then:

\[
e=-5
\]

---

# 24. Why square the error?

Without squaring:

```text
+5 + (-5) = 0
```

Two bad predictions could cancel.

With squared error:

\[
5^2+(-5)^2=25+25=50
\]

So all deviations contribute positively.

---

# 25. Why use `1/(2m)`?

The division by \(m\) averages across samples.

The factor \(1/2\) simplifies differentiation.

Because:

\[
\frac{d}{de}e^2=2e
\]

the factor `2` cancels with the denominator `2`.

---

# 26. Is this exactly MSE?

No.

Ordinary MSE is:

\[
MSE=\frac{1}{m}\sum e_i^2
\]

Our cost is:

\[
J=\frac{1}{2m}\sum e_i^2
\]

Therefore:

\[
\boxed{MSE=2J}
\]

That distinction is important in viva.

---

# 27. Gradient descent

The core algorithm is:

```python
predictions = X_train.dot(theta)
errors = predictions - y_train
gradient = (1 / m) * X_train.T.dot(errors)
theta = theta - learning_rate * gradient
```

Mathematically:

\[
\boxed{
\nabla J(\theta)=
\frac{1}{m}X^T(X\theta-y)
}
\]

and:

\[
\boxed{
\theta \leftarrow
\theta-\alpha\nabla J(\theta)
}
\]

where \(\alpha\) is the learning rate.

---

# 28. Why is gradient descent almost unchanged from Lab 01?

The vectorized formula works regardless of how many features there are.

Lab 01:

```text
X has 2 columns:
[1, x]
```

Lab 02:

```text
X has 5 columns:
[1, AT, V, AP, RH]
```

But:

```python
X.dot(theta)
```

and:

```python
X.T.dot(errors)
```

still work.

This is one major advantage of vectorized matrix notation.

---

# 29. Dimensions in Part A

After adding the dummy feature:

```text
X_train.shape = (7655, 5)
theta.shape   = (5,)
y_train.shape = (7655,)
```

Prediction:

```text
(7655,5) dot (5,)
        ↓
     (7655,)
```

Gradient:

```text
X.T             = (5,7655)
errors          = (7655,)
X.T dot errors  = (5,)
```

Therefore the gradient contains one value for every parameter.

---

# 30. What does the gradient mean?

For five parameters:

```text
gradient[0] -> change direction for theta0
gradient[1] -> change direction for AT coefficient
gradient[2] -> change direction for V coefficient
gradient[3] -> change direction for AP coefficient
gradient[4] -> change direction for RH coefficient
```

---

# 31. Training error and validation error

The notebook records both:

```python
train_cost_history.append(train_cost)
val_cost_history.append(val_cost)
```

Therefore every iteration gives:

```text
iteration 1  -> training cost, validation cost
iteration 2  -> training cost, validation cost
iteration 3  -> training cost, validation cost
...
```

This allows two curves to be plotted together.

---

# 32. Why do we save `best_theta`?

The model at the final iteration is not automatically the best validation model.

During training:

```python
if val_cost < best_val_cost:
    best_val_cost = val_cost
    best_theta = theta.copy()
    best_iteration = i + 1
```

So the notebook remembers the parameter set associated with the lowest validation cost.

---

# 33. Why `theta.copy()`?

If we only stored a reference to the mutable array and later continued updating it, the saved "best" parameters could change.

`copy()` stores the actual values at that moment.

---

# 34. What is `np.inf` doing?

```python
best_val_cost = np.inf
```

`np.inf` means positive infinity.

Any real validation cost will be smaller than infinity.

Therefore the first valid model automatically becomes the current best model.

---

# 35. `train()` function

`train()` is a wrapper:

```python
n = X_train.shape[1]
theta = np.zeros(n)
```

Then it calls gradient descent.

This keeps the same structure as Lab 01:

```text
train()
   |
   +--> initialize theta
   |
   +--> gradient_descent()
```

That makes the implementation modular.

---

# 36. Why initialize theta with zero?

For linear regression, zero initialization is valid.

Example:

```text
theta = [0, 0, 0, 0, 0]
```

Initially:

\[
\hat y=0
\]

The first gradient is nonzero, so parameters begin moving toward better values.

Unlike some neural-network settings, symmetry is not a problem here.

---

# 37. Learning rate

Learning rate controls the step size:

\[
\theta\leftarrow\theta-\alpha\nabla J
\]

Small \(\alpha\):

```text
small updates
-> stable but slow convergence
```

Large \(\alpha\):

```text
large updates
-> faster if appropriate
-> may overshoot/diverge if too large
```

---

# 38. Part A — without scaling

The notebook used:

```text
learning rate = 5e-7
iterations    = 5000
```

Result:

```text
Best training cost   = 54.835725
Best validation cost = 54.837414
Best iteration       = 5000
```

The unscaled features require a very small learning rate.

---

# 39. Why does the unscaled model need such a tiny learning rate?

Feature magnitudes are very different.

Rough examples:

```text
AT ≈ tens
V  ≈ tens
AP ≈ around 1000
RH ≈ tens
```

`AP` is numerically much larger than several other variables.

Gradient components can therefore have very different magnitudes.

A learning rate that is safe for one direction may be too large for another.

---

# 40. Part A — with scaling

Scaled experiment:

```text
learning rate = 0.05
iterations    = 1000
```

Result:

```text
Best training cost   = 10.482488
Best validation cost = 10.007414
Best iteration       = 1000
```

This demonstrates that scaling allows much larger and more effective gradient-descent updates.

---

# 41. Does feature scaling theoretically change the best linear model?

If optimization fully converges, linear regression on standardized features and linear regression on original features represent the same class of linear functions.

Scaling mainly changes:

- numerical conditioning,
- gradient magnitudes,
- suitable learning rate,
- convergence speed.

In this lab, because gradient descent is run for a finite number of iterations, the scaled version reaches a much better solution within the available training budget.

---

# 42. Learned parameters for the scaled Part A model

Working/standardized-coordinate parameters:

```text
theta_0      = 454.248138
theta_AT     = -14.543817
theta_V      = -3.114429
theta_AP     = 0.428379
theta_RH     = -2.263662
```

These coefficients multiply standardized features, not the original raw feature values.

---

# 43. Converting parameters back to original scale

If:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}
\]

and:

\[
\hat y=
\theta_0+\sum_j\theta_jz_j
\]

then:

\[
\hat y=
\theta_0+
\sum_j
\theta_j
\frac{x_j-\mu_j}{\sigma_j}
\]

Expanding:

\[
\hat y=
\theta_0-
\sum_j\frac{\theta_j\mu_j}{\sigma_j}
+
\sum_j\frac{\theta_j}{\sigma_j}x_j
\]

Therefore:

\[
\boxed{
\beta_j=\frac{\theta_j}{\sigma_j}
}
\]

and:

\[
\boxed{
\beta_0=
\theta_0-
\sum_j
\frac{\theta_j\mu_j}{\sigma_j}
}
\]

---

# 44. Original-scale Part A equation

The notebook obtained approximately:

```text
Intercept         = 443.666362
Coefficient of AT = -1.951346
Coefficient of V  = -0.245150
Coefficient of AP =  0.072762
Coefficient of RH = -0.154903
```

So:

\[
\boxed{
\widehat{PE}
=
443.666362
-1.951346AT
-0.245150V
+0.072762AP
-0.154903RH
}
\]

---

# 45. How do we interpret a multiple-regression coefficient?

Example:

```text
AT coefficient = -1.951346
```

Correct interpretation:

> Holding the other input variables fixed, increasing AT by one unit is associated with approximately a 1.95-unit decrease in predicted PE.

The phrase **holding the other variables fixed** is important.

This is different from simple regression.

---

# 46. Why can't we interpret `theta_AT = -14.54` as a one-degree effect?

Because `-14.54` belongs to standardized AT.

One unit of standardized AT means:

```text
one standard deviation of AT
```

not one original degree.

For original-unit interpretation, use the converted coefficient:

```text
-1.951346
```

---

# 47. Training and validation error curves

Both curves should generally fall as optimization improves.

A typical shape:

```text
Cost
|\
| \
|  \
|   \________ training
|    \_______ validation
+----------------------> iterations
```

If validation error starts increasing while training error continues falling, that may indicate overfitting.

---

# 48. Part B — What is k-fold cross validation?

K-fold cross validation divides the dataset into \(k\) approximately equal parts.

For \(k=5\):

```text
Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5
```

Training occurs five times.

---

# 49. 5-fold process

```text
Run 1:
Validation = Fold 1
Training   = Folds 2,3,4,5

Run 2:
Validation = Fold 2
Training   = Folds 1,3,4,5

Run 3:
Validation = Fold 3
Training   = Folds 1,2,4,5

Run 4:
Validation = Fold 4
Training   = Folds 1,2,3,5

Run 5:
Validation = Fold 5
Training   = Folds 1,2,3,4
```

Every example is used:

- for training in 4 runs,
- for validation in 1 run.

---

# 50. Why use cross validation?

A single train-validation split can be lucky or unlucky.

Its result depends on which particular samples land in validation.

Cross validation evaluates the model across multiple partitions.

Therefore the evaluation is less dependent on one random split.

---

# 51. `create_k_folds()`

```python
indices = rng.permutation(n_samples)
folds = np.array_split(indices, k)
```

First:

```text
shuffle sample indices
```

Then:

```text
split them into k groups
```

`np.array_split()` is useful because the number of examples does not have to be perfectly divisible by `k`.

---

# 52. Scaling inside each fold — VERY important

For every fold:

```text
current training folds
        |
        v
calculate mean/std
        |
        +------> scale current training data
        |
        +------> scale current validation fold
```

We do **not** calculate one global mean/std before cross validation.

That would leak validation information into training.

---

# 53. Cross-validation results

Average validation cost:

\[
\boxed{10.391403}
\]

Average original-scale parameters:

```text
Intercept ≈ 449.068297
AT        ≈ -1.955404
V         ≈ -0.243505
AP        ≈  0.067370
RH        ≈ -0.154377
```

---

# 54. Why are parameters slightly different in every fold?

Each run trains on a slightly different subset.

Therefore:

- means/stds are slightly different,
- training examples are slightly different,
- the best iteration may differ,
- learned coefficients may differ.

This is expected.

---

# 55. Single split vs 5-fold parameters

Single split:

```text
Intercept 443.666362
AT        -1.951346
V         -0.245150
AP         0.072762
RH        -0.154903
```

5-fold average:

```text
Intercept 449.068297
AT        -1.955404
V         -0.243505
AP         0.067370
RH        -0.154377
```

Most feature coefficients are very close.

That suggests the learned relationship is reasonably stable across partitions.

---

# 56. Why isn't the cross-validation result simply one final model?

Cross validation is primarily an **evaluation procedure**.

We obtain several models to estimate how the learning procedure behaves.

A common practical workflow is:

```text
cross-validation
        ↓
choose hyperparameters/model settings
        ↓
retrain final model on available training data
```

In this lab, the task mainly asks us to compare parameters and validation performance.

---

# 57. Hyperparameters vs parameters

This distinction is a common viva question.

## Parameters

Learned from data:

```text
theta0
theta1
theta2
theta3
theta4
```

## Hyperparameters

Chosen by us:

```text
learning rate
number of iterations
k in k-fold CV
polynomial degree d
```

Degree \(d\) is therefore a hyperparameter.

---

# 58. Part C — Polynomial regression

A linear model with one variable is:

\[
\hat y=\theta_0+\theta_1x
\]

Polynomial regression extends the input representation.

Degree 2:

\[
\hat y=\theta_0+\theta_1x+\theta_2x^2
\]

Degree 3:

\[
\hat y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3
\]

---

# 59. Is polynomial regression still linear regression?

Yes — **linear in the parameters**.

For degree 3:

\[
\hat y=
\theta_0+
\theta_1x+
\theta_2x^2+
\theta_3x^3
\]

The model is nonlinear in the original variable \(x\), but each parameter appears linearly.

Therefore we can still use the same linear-regression gradient-descent machinery.

---

# 60. Why is this powerful?

We transform one original feature:

```text
x
```

into several features:

```text
x
x²
x³
```

Then ordinary linear regression in this expanded feature space can represent curved functions in the original x-axis.

---

# 61. Polynomial feature scaling

The notebook first standardizes the original \(x\):

\[
z=\frac{x-\mu}{\sigma}
\]

Then creates polynomial columns.

For degree 1:

```text
[1, z]
```

Degree 2:

```text
[1, z, z²]
```

Degree 3:

```text
[1, z, z², z³]
```

---

# 62. Why scale before creating higher powers?

Without scaling, values such as:

\[
x^2,\quad x^3
\]

can become much larger than \(x\).

That can make gradient descent poorly conditioned or unstable.

Using standardized \(z\) keeps polynomial values in a more manageable numerical range.

---

# 63. `process_polynomial_data()`

The key code is conceptually:

```python
z_train = (x_train - mean) / std

X_train = np.column_stack([
    z_train ** power
    for power in range(degree + 1)
])
```

If `degree = 3`, then:

```python
range(4)
```

gives:

```text
0,1,2,3
```

so the columns are:

```text
z^0 = 1
z^1 = z
z^2
z^3
```

The \(z^0\) column automatically becomes the dummy-feature column.

---

# 64. Why does `z**0` give the intercept column?

For every nonzero value:

\[
z^0=1
\]

NumPy also produces 1 for zero to the zero power in this context.

Thus the first column is all ones.

---

# 65. Training degree 1, 2 and 3

The notebook uses the same:

```text
train()
gradient_descent()
compute_cost()
```

functions.

Only the design matrix changes.

This demonstrates modularity.

The regression algorithm does not need separate implementations for d=1, d=2, d=3.

---

# 66. Validation results for polynomial degree

The notebook obtained approximately:

```text
d = 1
Training cost   = 14.951499
Validation cost = 13.773533
Best iteration  = 389

d = 2
Training cost   = 13.788425
Validation cost = 12.453008
Best iteration  = 691

d = 3
Training cost   = 13.085323
Validation cost = 11.840282
```

Therefore:

\[
\boxed{d=3}
\]

had the lowest validation error among the tested degrees.

---

# 67. Why choose degree using validation error instead of training error?

Training error usually favors a more flexible model.

A more complex model can fit its training data increasingly well.

But our goal is not merely:

```text
memorize training data
```

Our goal is:

```text
generalize to unseen data
```

Therefore degree selection should be based on validation performance.

---

# 68. Underfitting

Underfitting happens when the model is too simple to represent the important relationship.

Possible signs:

```text
high training error
high validation error
```

For example, if the true relationship is strongly curved, degree 1 may underfit.

---

# 69. Overfitting

Overfitting happens when the model fits training data too specifically, including noise.

Typical pattern:

```text
very low training error
higher validation error
```

A higher polynomial degree does not always mean a better model.

In this lab we only compare degrees 1, 2 and 3.

Degree 3 happened to give the lowest validation error among these options.

That does **not** prove that arbitrarily larger degrees would keep improving the model.

---

# 70. Best polynomial model

For degree 3, standardized-coordinate parameters were approximately:

```text
theta_0 = 452.454297
theta_1 = -18.008815
theta_2 =   1.898507
theta_3 =   1.083787
```

These are coefficients for powers of standardized \(z\).

---

# 71. Original-scale polynomial equation

The notebook converts the polynomial back to powers of original \(x\):

```text
a0 = 493.308918
a1 = -0.713772
a2 = -0.120571
a3 =  0.002618
```

Therefore:

\[
\boxed{
\hat y
\approx
493.308918
-0.713772x
-0.120571x^2
+0.002618x^3
}
\]

---

# 72. Why is polynomial coefficient conversion more complicated?

For one standardized variable:

\[
z=\frac{x-\mu}{\sigma}
\]

A cubic model is:

\[
\theta_0+\theta_1z+\theta_2z^2+\theta_3z^3
\]

But:

\[
z^2=
\left(\frac{x-\mu}{\sigma}\right)^2
\]

and:

\[
z^3=
\left(\frac{x-\mu}{\sigma}\right)^3
\]

Expanding these produces mixtures of:

```text
constant
x
x²
x³
```

Therefore every standardized coefficient can contribute to several original-scale coefficients.

The notebook uses `np.polynomial.Polynomial` to perform this algebra safely.

---

# 73. Why plot all three polynomial curves together?

It lets us visually compare model flexibility.

Conceptually:

```text
d=1 -> straight line
d=2 -> one quadratic bend
d=3 -> cubic shape
```

The graph helps us see how higher-degree models capture additional curvature.

But the final choice is made using validation error, not visual appearance alone.

---

# 74. Why make a validation-error bar chart?

The bar chart gives a direct model-selection comparison.

```text
Validation Error
|
| █
| █   █
| █   █   █
+-------------
  1   2   3
```

The shortest bar corresponds to the lowest validation cost.

In this run, degree 3 has the lowest bar.

---

# 75. Why plot error curves separately for every d?

Because convergence behavior can differ with model complexity.

For each degree, we want to inspect:

```text
training cost vs iteration
validation cost vs iteration
```

This can reveal:

- convergence speed,
- possible divergence,
- best iteration,
- gap between training and validation performance.

---

# 76. One formula set you MUST memorize

## Prediction

\[
\boxed{\hat y=X\theta}
\]

## Error

\[
\boxed{e=X\theta-y}
\]

## Cost

\[
\boxed{
J(\theta)
=
\frac{1}{2m}
\sum e^2
}
\]

## Gradient

\[
\boxed{
\nabla J(\theta)
=
\frac{1}{m}X^Te
}
\]

## Update

\[
\boxed{
\theta
\leftarrow
\theta-\alpha\nabla J(\theta)
}
\]

## Standardization

\[
\boxed{
z=
\frac{x-\mu}{\sigma}
}
\]

These equations explain most of the code.

---

# 77. Lab 01 vs Lab 02

| Concept | Lab 01 | Lab 02 |
|---|---|---|
| Regression | Simple linear | Multiple + polynomial |
| Features | 1 | 4 in Part A |
| Validation | Mainly training/error experiment | Explicit validation |
| Scaling | One feature | Multiple features |
| Model selection | Learning-rate experiment | Validation + polynomial degree |
| Cross validation | No | 5-fold |
| Polynomial features | No | d = 1, 2, 3 |
| Core gradient formula | Same | Same |

The most important point:

> Lab 02 does not replace Lab 01. It generalizes the same vectorized regression code.

---

# 78. Results you should remember

## Part A

Without feature scaling:

```text
Best validation cost ≈ 54.837414
Best training cost   ≈ 54.835725
Learning rate        = 5e-7
Iterations           = 5000
```

With feature scaling:

```text
Best validation cost ≈ 10.007414
Training cost        ≈ 10.482488
Learning rate        = 0.05
Iterations           = 1000
```

Original-scale model:

\[
\widehat{PE}
\approx
443.666
-1.951AT
-0.245V
+0.0728AP
-0.1549RH
\]

---

## Part B

```text
5-fold average validation cost ≈ 10.391403
```

Average coefficients:

```text
Intercept ≈ 449.068
AT        ≈ -1.9554
V         ≈ -0.2435
AP        ≈  0.06737
RH        ≈ -0.15438
```

---

## Part C

```text
d=1 validation cost ≈ 13.7735
d=2 validation cost ≈ 12.4530
d=3 validation cost ≈ 11.8403
```

Best:

\[
\boxed{d=3}
\]

Best original-scale equation:

\[
\hat y
\approx
493.309
-0.7138x
-0.1206x^2
+0.002618x^3
\]

---

# 79. How to explain the entire Lab 02 to Sir

A concise strong answer:

> "Lab 02 extends our Lab 01 linear-regression implementation. First, for the power-plant dataset I use four features — AT, V, AP and RH — to predict PE. I randomly split the data into training and validation sets. For feature scaling, I compute each mean and standard deviation only from the training set and apply the same transformation to validation data to avoid leakage. I add a dummy feature for the intercept and train using the same vectorized batch gradient descent as Lab 01. At every iteration I store both training and validation cost and retain the parameters with the lowest validation cost. I compare optimization with and without scaling. Then I implement manual 5-fold cross validation, fitting the scaler independently inside each training fold. Finally, for polynomial regression I standardize one x feature, create powers up to d=1,2,3, use validation error to select the degree, and degree 3 gives the lowest validation error."

---

# 80. Viva Q&A — Basics

### Q1. What is multiple linear regression?

Multiple linear regression predicts a continuous target using more than one input feature:

\[
\hat y=\theta_0+\theta_1x_1+\cdots+\theta_nx_n
\]

---

### Q2. How is it different from simple linear regression?

Simple linear regression has one input feature. Multiple linear regression has two or more input features.

---

### Q3. What are the four features in this lab?

`AT`, `V`, `AP`, and `RH`.

---

### Q4. What is the target?

`PE`, the net electrical energy output.

---

### Q5. How many examples are in the dataset?

9568.

---

### Q6. How many parameters does the Part A model learn?

Five:

```text
theta0
theta_AT
theta_V
theta_AP
theta_RH
```

One intercept and four feature coefficients.

---

### Q7. Why does X have five columns when there are only four original features?

Because the first column is the dummy feature \(x_0=1\) used for the intercept.

---

### Q8. What does `X.dot(theta)` calculate?

Predictions for all examples.

---

### Q9. What type of machine learning is this?

Supervised learning.

---

### Q10. Why?

Because during training we know both the features and the correct target values.

---

# 81. Viva Q&A — Training and validation

### Q11. What is a training set?

The subset used to calculate gradients and update model parameters.

---

### Q12. What is a validation set?

A separate subset used to evaluate model performance during model/hyperparameter selection.

---

### Q13. Does validation data update theta?

No.

---

### Q14. Then why calculate validation error every iteration?

To see how well the current model generalizes and to identify the best parameter state.

---

### Q15. Why shuffle before splitting?

To avoid a potentially biased split caused by original data ordering.

---

### Q16. Why use a seed?

For reproducibility.

---

### Q17. Why 80/20?

It is a common practical split that leaves most data for training while reserving enough examples for validation. It is not a universal mathematical rule.

---

# 82. Viva Q&A — Feature scaling

### Q18. Which feature scaling method did you use?

Standardization:

\[
z=\frac{x-\mu}{\sigma}
\]

---

### Q19. Why standardize?

To place features on comparable numerical scales and make gradient descent converge more stably and quickly.

---

### Q20. Does scaling mean every feature becomes between 0 and 1?

No. That would be min-max normalization.

Standardization gives approximately mean 0 and standard deviation 1.

---

### Q21. Why does AP create a problem compared with AT?

AP has a much larger numerical magnitude, so without scaling different gradient directions have very different scales.

---

### Q22. Do you calculate validation mean/std separately?

No.

The validation data must use the mean and standard deviation calculated from training data.

---

### Q23. Why?

Because at deployment time unseen data does not get to redefine the training preprocessing. Also, using validation statistics would leak information.

---

### Q24. What is data leakage?

When information from validation/test data influences model training or preprocessing in a way that should not be available during training.

---

### Q25. Why calculate scaling separately inside each CV fold?

Because the current validation fold must remain unseen. Each fold's scaler must be fit only on its own training subset.

---

# 83. Viva Q&A — Gradient descent

### Q26. What is gradient descent?

An iterative optimization algorithm that updates parameters in the direction that reduces the cost.

---

### Q27. What is the gradient formula?

\[
\frac{1}{m}X^T(X\theta-y)
\]

---

### Q28. Why do we subtract the gradient?

Because the gradient points toward increasing cost; subtracting moves toward decreasing cost.

---

### Q29. What is the learning rate?

The step size of each gradient-descent update.

---

### Q30. What happens if learning rate is too large?

The algorithm may overshoot the minimum, oscillate, or diverge.

---

### Q31. What happens if learning rate is too small?

Training may be stable but unnecessarily slow.

---

### Q32. Why could the scaled experiment use 0.05 while unscaled used 5e-7?

Scaling makes the numerical ranges and gradient magnitudes much better conditioned.

---

### Q33. Is this batch, stochastic, or mini-batch gradient descent?

Batch gradient descent.

The gradient uses every training example in each iteration.

---

### Q34. Where exactly is batch behavior visible?

```python
gradient = (1 / m) * X_train.T.dot(errors)
```

`errors` contains errors for the full training set.

---

# 84. Viva Q&A — Cost

### Q35. What cost function did you use?

\[
J=\frac{1}{2m}\sum(\hat y-y)^2
\]

---

### Q36. Why square errors?

To prevent positive and negative errors from cancelling and to penalize larger deviations more strongly.

---

### Q37. Why the factor 1/2?

It simplifies the derivative.

---

### Q38. Is this exactly MSE?

No. It is half the usual MSE.

---

### Q39. Why track training and validation costs separately?

Training cost measures fit to learned data. Validation cost measures generalization to unseen data.

---

### Q40. Which cost do you use to save the best model?

Validation cost.

---

# 85. Viva Q&A — Best model logic

### Q41. Why not just use final theta?

The lowest validation cost may occur before the final iteration.

---

### Q42. What does `best_iteration` mean?

The iteration at which the smallest validation cost was observed.

---

### Q43. What does `best_theta` contain?

A copy of the parameter values from the best validation iteration.

---

### Q44. Why use `np.inf` initially?

So that the first real validation cost is guaranteed to be smaller.

---

### Q45. Why use `theta.copy()`?

To preserve the parameter values at that iteration rather than referencing an array that continues changing.

---

# 86. Viva Q&A — Multiple-regression coefficients

### Q46. What does a negative AT coefficient mean?

Holding the other variables fixed, increasing AT is associated with a decrease in predicted PE.

---

### Q47. Why must you say "holding other variables fixed"?

Because multiple regression estimates each feature's coefficient while the remaining features are included in the model.

---

### Q48. Can you say AT causes PE to decrease?

Not from regression alone. The model demonstrates an association/predictive relationship, not necessarily causation.

---

### Q49. Why are standardized coefficients different from original-scale coefficients?

They correspond to different units. Standardized coefficients multiply z-scores; original coefficients multiply raw feature values.

---

### Q50. Which coefficients are easier to interpret physically?

Original-scale coefficients.

---

# 87. Viva Q&A — Cross validation

### Q51. What does 5-fold cross validation mean?

Split the data into five subsets and train five models, using each subset once as validation.

---

### Q52. How many times is each sample used for validation?

Exactly once.

---

### Q53. How many times is each sample used for training?

Four times.

---

### Q54. Why is cross validation better than one split?

It reduces dependence on one particular random train-validation partition.

---

### Q55. Does cross validation guarantee perfect evaluation?

No. It gives a more robust estimate, but results still depend on the data, model and procedure.

---

### Q56. Why do coefficients differ between folds?

Because each fold trains on a different subset.

---

### Q57. Why average the fold coefficients?

To summarize how the learned parameters behave across multiple training partitions.

---

### Q58. What is the average 5-fold validation cost in your run?

Approximately 10.3914.

---

### Q59. Is k=5 a model parameter?

No. It is a hyperparameter/evaluation setting.

---

### Q60. What would happen if k=n?

That becomes leave-one-out cross validation: each validation fold contains one sample.

---

# 88. Viva Q&A — Polynomial regression

### Q61. What is polynomial regression?

Linear regression performed on polynomial transformations of the input feature, such as \(x,x^2,x^3\).

---

### Q62. Why is polynomial regression still considered linear regression?

Because the prediction is linear with respect to the learned coefficients.

---

### Q63. What does degree 1 mean?

\[
\hat y=\theta_0+\theta_1x
\]

A straight line.

---

### Q64. What does degree 2 mean?

\[
\hat y=\theta_0+\theta_1x+\theta_2x^2
\]

A quadratic model.

---

### Q65. What does degree 3 mean?

\[
\hat y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3
\]

A cubic model.

---

### Q66. Why scale x before generating polynomial powers?

To avoid very large values in \(x^2\) and \(x^3\) and improve numerical stability during gradient descent.

---

### Q67. Which degree performed best?

Degree 3.

---

### Q68. Why did you choose d=3?

It produced the lowest validation cost among d=1,2,3.

---

### Q69. Why not choose based on training cost?

Because training error can favor unnecessarily complex models and does not directly measure performance on unseen data.

---

### Q70. Does degree 3 being best mean degree 10 must be even better?

No.

Higher degrees may overfit and have worse validation error.

---

# 89. Viva Q&A — Underfitting and overfitting

### Q71. What is underfitting?

When a model is too simple to capture the important structure in the data.

---

### Q72. What is overfitting?

When a model fits training data too specifically and performs worse on unseen data.

---

### Q73. How can training and validation curves indicate overfitting?

Training error may continue decreasing while validation error starts increasing.

---

### Q74. Could d=1 underfit compared with d=3?

Yes, if the underlying relationship contains curvature that a straight line cannot capture.

---

### Q75. Is d=3 definitely the true mathematical relationship?

No. It is only the best of the tested degrees according to this validation split.

---

# 90. Viva Q&A — Code-specific questions

### Q76. What does `axis=0` mean in `np.mean(X_train, axis=0)`?

Compute one mean for each feature column across all training rows.

---

### Q77. What does `X[:, j]` mean?

Take every row from column `j`.

---

### Q78. What does `np.column_stack()` do?

Combines arrays as columns of one matrix.

---

### Q79. What does `np.setdiff1d(all_indices, validation_indices)` do?

Returns the indices not present in the validation fold, which become training indices.

---

### Q80. What does `np.vstack(original_thetas)` do?

Stacks parameter vectors vertically to form a matrix so we can average each parameter across folds.

---

### Q81. What does `np.mean(..., axis=0)` do there?

Calculates one average for each parameter across all fold models.

---

### Q82. Why return `scale_params`?

Because we need the training mean/std later for validation preprocessing, plotting, and converting parameters back to original scale.

---

### Q83. Why do polynomial powers start from zero?

Because \(z^0=1\), which automatically creates the intercept/dummy column.

---

### Q84. Why use `np.linspace()` for the fitted polynomial plot?

It generates ordered, evenly spaced x-values so the fitted model can be drawn as a smooth curve.

---

# 91. Trick viva questions

### Q85. If I remove the dummy feature, what happens?

The model no longer has a separate intercept unless the algorithm is redesigned. It would effectively be forced through the origin in feature space.

---

### Q86. What happens if standard deviation is zero?

Division by zero would occur during standardization. The code protects against this by replacing zero standard deviations with 1.

---

### Q87. If validation cost is lower than training cost, is that impossible?

No. A randomly selected validation subset can happen to be slightly easier than the training subset.

In this run, the scaled validation cost is slightly lower than training cost.

---

### Q88. Why is the unscaled validation cost much worse? Is scaling changing the dataset?

Scaling is an invertible transformation for nonconstant features. The main issue is optimization: the finite-step gradient descent converges much more effectively after scaling.

---

### Q89. Could we solve linear regression without gradient descent?

Yes. For standard linear regression, a closed-form solution such as the normal equation/pseudoinverse can be used. But this lab specifically practices gradient-descent implementation.

---

### Q90. Why not use `sklearn.LinearRegression`?

The purpose of this lab is to implement and understand the algorithm manually: cost, gradient descent, preprocessing and validation.

---

### Q91. If we used sklearn, would the theory disappear?

No. The same regression principles still apply, but the low-level optimization would be hidden inside the library.

---

### Q92. If I multiply one feature by 100, will the meaning of predictions necessarily change?

If the corresponding coefficient is adjusted appropriately, the same function can be represented. But optimization behavior under gradient descent can change greatly without scaling.

---

### Q93. If learning rate is zero, what happens?

Theta never changes.

---

### Q94. If iterations are zero, what happens?

No gradient-descent update occurs and theta remains at its initialization.

---

### Q95. Why do we use the same seed in experiments?

To make comparisons reproducible rather than mixing algorithm differences with random-split differences.

---

# 92. Questions directly about your output

### Q96. What was your validation error without scaling?

Approximately:

```text
54.8374
```

---

### Q97. What was your validation error with scaling?

Approximately:

```text
10.0074
```

---

### Q98. What does that experiment demonstrate?

Feature scaling substantially improves gradient-descent convergence for this dataset and chosen training budget.

---

### Q99. What was your 5-fold average validation cost?

Approximately:

```text
10.3914
```

---

### Q100. Which polynomial degree was selected?

```text
d = 3
```

---

### Q101. What was its validation cost?

Approximately:

```text
11.8403
```

---

### Q102. What are the original-scale cubic coefficients?

Approximately:

```text
a0 = 493.308918
a1 = -0.713772
a2 = -0.120571
a3 =  0.002618
```

---

# 93. If Sir asks: "Show me exactly where prediction happens"

Answer:

```python
predictions = X_train.dot(theta)
```

This implements:

\[
\hat y=X\theta
\]

---

# 94. If Sir asks: "Show me exactly where error happens"

Answer:

```python
errors = predictions - y_train
```

---

# 95. If Sir asks: "Show me exactly where gradient happens"

Answer:

```python
gradient = (1 / m) * X_train.T.dot(errors)
```

---

# 96. If Sir asks: "Show me exactly where parameters are updated"

Answer:

```python
theta = theta - learning_rate * gradient
```

---

# 97. If Sir asks: "Show me where validation selects the best model"

Answer:

```python
if val_cost < best_val_cost:
    best_val_cost = val_cost
    best_theta = theta.copy()
    best_iteration = i + 1
```

---

# 98. If Sir asks: "Show me where leakage is prevented"

For a normal split:

```python
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_processed = (X_train - mean) / std
X_val_processed = (X_val - mean) / std
```

For cross validation, the same processing function is called **inside each fold** using only that fold's training subset.

---

# 99. If Sir asks: "Where are the polynomial features created?"

Answer:

```python
X_train = np.column_stack([
    z_train ** power
    for power in range(degree + 1)
])
```

For degree 3, this produces:

```text
[1, z, z², z³]
```

---

# 100. If Sir asks: "What is the most important idea of this lab?"

A strong answer:

> "The main idea is that the same vectorized linear-regression machinery can be generalized. We can add more input columns for multiple regression, repeatedly change the training/validation partition for cross validation, and transform one x into polynomial columns for nonlinear curve fitting. The core prediction, cost, gradient and update equations remain the same."

---

# 101. Final revision sheet

Memorize this:

```text
PART A
------
Features: AT, V, AP, RH
Target: PE

Prediction:
y_hat = X theta

Cost:
J = 1/(2m) * sum((y_hat-y)^2)

Gradient:
1/m * X.T * (X theta-y)

Update:
theta = theta - alpha*gradient

Scaling:
z = (x-mean)/std

Important:
mean/std come ONLY from training data.

Best scaled validation cost ≈ 10.0074


PART B
------
5-fold CV:
train 5 times
each fold is validation once
scale separately inside each fold

Average validation cost ≈ 10.3914


PART C
------
d=1 -> [1,z]
d=2 -> [1,z,z²]
d=3 -> [1,z,z²,z³]

Choose degree using validation error.

Best d = 3
Best validation cost ≈ 11.8403
```

---

# 102. The ten most likely questions to prepare first

1. **Why do we need a validation set?**
2. **Why must scaling use only training statistics?**
3. **Why does feature scaling help gradient descent?**
4. **What is the gradient formula and where is it implemented?**
5. **What is the difference between parameter and hyperparameter?**
6. **How does 5-fold cross validation work?**
7. **Why scale independently inside every CV fold?**
8. **Why is polynomial regression still linear regression?**
9. **Why choose polynomial degree using validation error instead of training error?**
10. **What is the difference between underfitting and overfitting?**

If you can answer those and explain the five equations below, you understand the core of Lab 02:

\[
\hat y=X\theta
\]

\[
e=X\theta-y
\]

\[
J=\frac{1}{2m}\sum e^2
\]

\[
\nabla J=\frac{1}{m}X^Te
\]

\[
\theta\leftarrow\theta-\alpha\nabla J
\]

---

# 103. Final one-minute explanation

> "In Lab 02 I extended my Lab 01 regression code. For multiple linear regression I use four power-plant features to predict PE, add a dummy feature, and train with vectorized batch gradient descent. I split the data into training and validation sets and standardize using only training statistics to prevent leakage. I keep both training and validation cost histories and save theta from the lowest validation error. Scaling significantly improves convergence. Then I manually implement 5-fold cross validation, where each fold becomes validation once and scaling is fitted independently inside every training fold. Finally, for polynomial regression I standardize x, transform it into powers up to degree 1, 2 and 3, reuse the same linear-regression algorithm, and select degree 3 because it has the lowest validation error."

