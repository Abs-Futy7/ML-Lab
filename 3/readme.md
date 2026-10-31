# Lab 03: Classification and the Perceptron

Yes. The easiest way to prepare for this lab is to see the whole topic as one chain:

$$
\boxed{\text{features } x \rightarrow \text{score } \theta^T x \rightarrow \text{decision boundary} \rightarrow \text{class}}
$$

Then the perceptron is simply the algorithm that learns the parameter vector $\theta$ so that the classifier separates the classes correctly.

This guide is meant to support the Lab 03 notebook and the official assignment sheet. The notebook includes a from-scratch perceptron implementation, linear and nonlinear decision boundaries, feature lifting, label encoding, plotting, and augmentation ideas.

**Download notebook:** `Lab3_Classification_Perceptron_Augmentation.ipynb`

Your attached official Lab 03 sheet says to finalize the project title, download a sample ML dataset (for example, from Mendeley Data), and apply data augmentation. The notebook is intended to help you understand the underlying machine learning idea behind that task.

---

## 1. What is classification?

In regression, the output is a continuous value.

Examples:

$$
\text{house price}=4.2\text{ million BDT}
$$

$$
\text{temperature}=31.4^\circ\text{C}
$$

In classification, the output belongs to a discrete class.

| Input | Classification output |
|---|---|
| Email | Spam / Not Spam |
| Medical image | Disease / Healthy |
| Transaction | Fraud / Legitimate |
| Image | Cat / Dog |
| Student information | Pass / Fail |

For binary classification, we often represent the output as either

$$
y\in\{0,1\}
$$

or

$$
y\in\{-1,+1\}
$$

For perceptron, the encoding $\{-1,+1\}$ is especially convenient.

---

## 2. Problem formulation

Suppose every example has two features:

$$
x=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

and the target label is:

$$
y\in\{-1,+1\}
$$

Our training dataset is:

$$
\mathcal D = \{(x^{(1)},y^{(1)}), \ldots, (x^{(m)},y^{(m)})\}
$$

The objective is to learn a function

$$
h_\theta(x)
$$

such that for a new unseen input $x$, the model predicts the correct class.

For a linear classifier, we first compute a score:

$$
z = \theta_0 + \theta_1x_1 + \theta_2x_2
$$

and classify according to the sign:

$$
h_\theta(x)=
\begin{cases}
+1, & z\ge 0 \\
-1, & z<0
\end{cases}
$$

So the classifier is essentially:

$$
\boxed{h_\theta(x)=\operatorname{sign}(\theta^Tx)}
$$

---

## 3. What is a decision boundary?

The classifier decides based on the sign of the score:

$$
z>0 \Rightarrow +1
$$

and

$$
z<0 \Rightarrow -1
$$

The boundary between these two regions is where the score is exactly zero:

$$
z=0
$$

Therefore, the decision boundary is:

$$
\boxed{\theta_0 + \theta_1x_1 + \theta_2x_2 = 0}
$$

For two features, this is a line.

Example:

$$
-3 + 2x_1 + x_2 = 0
$$

or

$$
x_2 = 3 - 2x_1
$$

Everything on one side is classified as one class and everything on the other side as the other class.

The line itself is not a class. It is the place where the classifier is exactly undecided:

$$
\theta^Tx = 0
$$

---

## 4. Hyperplane vs decision boundary

These are closely related.

For a $d$-dimensional input,

$$
x=(x_1,x_2,\ldots,x_d)
$$

the equation

$$
\theta_0 + \theta_1x_1 + \cdots + \theta_dx_d = 0
$$

defines a hyperplane.

Its name depends on dimension:

| Input dimension | Boundary |
|---:|---|
| 1D | Point |
| 2D | Line |
| 3D | Plane |
| $d$D | Hyperplane |

The decision boundary of a linear classifier is a hyperplane.

So if the teacher asks:

> What is the distinction between a hyperplane and a decision boundary?

You can say:

> A hyperplane is a geometric object. In a linear classifier, the hyperplane $\theta^Tx=0$ acts as the decision boundary that separates the classes.

---

## 5. What do the parameters mean?

Consider:

$$
\theta_0 + \theta_1x_1 + \theta_2x_2 = 0
$$

The vector

$$
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}
$$

is perpendicular to the decision boundary. It controls the orientation of the line.

Meanwhile, $\theta_0$ moves the line away from the origin.

Example:

$$
x_1 + x_2 = 0
$$

passes through the origin, but

$$
x_1 + x_2 - 5 = 0
$$

does not. The constant $-5$ is the bias or intercept.

---

## 6. Why add a constant feature?

This is a common trick used in linear models.

Suppose

$$
z = \theta_0 + \theta_1x_1 + \theta_2x_2
$$

Define a new feature:

$$
x_0 = 1
$$

Then the feature vector becomes:

$$
x=
\begin{bmatrix}
1 \\
x_1 \\
x_2
\end{bmatrix}
$$

and the parameter vector becomes:

$$
\theta=
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2
\end{bmatrix}
$$

Now:

$$
\theta^Tx = \theta_0(1)+\theta_1x_1+\theta_2x_2
$$

so we can write simply:

$$
\boxed{z = \theta^Tx}
$$

This lets us avoid separate notation like $w^Tx+b$ and use one unified representation.

---

## 7. Why is this called "lifting"?

Suppose the original data point is:

$$
(x_1,x_2)
$$

We transform it into:

$$
(1,x_1,x_2)
$$

which is a point in a higher-dimensional space.

For example,

$$
(2,4)
$$

becomes

$$
(1,2,4)
$$

We have lifted the data from 2D into 3D.

Every lifted point has first coordinate $1$, so all such points live on the plane:

$$
x_0=1
$$

This allows us to rewrite an affine boundary such as

$$
2x_1 + x_2 - 3 = 0
$$
as

$$
\begin{bmatrix}
-3 & 2 & 1
\end{bmatrix}
\begin{bmatrix}
1 \\
x_1 \\
x_2
\end{bmatrix}
=0
$$

or simply

$$
\boxed{\theta^Tx=0}
$$

This is a common viva or exam concept.

---

## 8. Target label encoding

Suppose the original labels are:

$$
\text{No}=0,\qquad\text{Yes}=1
$$

Perceptron commonly encodes them as:

$$
0\rightarrow -1,
\qquad
1\rightarrow +1
$$

Why?

Because then the expression

$$
\boxed{y(\theta^Tx)}
$$

tells us whether the prediction is correct.

If true label is $y=+1$ and score is $\theta^Tx=5$, then:

$$
y(\theta^Tx)= (+1)(5)=5>0
$$

This is correct.

If true label is $y=-1$ and score is $\theta^Tx=-3$, then:

$$
(-1)(-3)=3>0
$$

This is also correct.

But if $y=+1$ and $\theta^Tx=-4$, then:

$$
(+1)(-4)=-4<0
$$

This is wrong.

Therefore:

$$
\boxed{y_i(\theta^Tx_i)>0 \Rightarrow \text{correct}}
$$

$$
\boxed{y_i(\theta^Tx_i)\le 0 \Rightarrow \text{incorrect or on the boundary}}
$$

This is a key reason why $\{-1,+1\}$ encoding is widely used in perceptron learning.

---

## 9. Relationship with linear regression

Linear regression uses:

$$
h_\theta(x)=\theta^Tx
$$

The perceptron also begins with a linear score:

$$
z=\theta^Tx
$$

But the interpretation is different.

| Linear Regression | Perceptron |
|---|---|
| Predict a continuous value | Predict a class |
| Output is $\theta^Tx$ | Output is $\operatorname{sign}(\theta^Tx)$ |
| Usually minimizes MSE | Uses perceptron-style classification loss |
| Every error affects parameters | Misclassified examples trigger updates |
| Regression | Classification |

You technically could take labels in $\{-1,+1\}$ and fit a linear regression model, then classify using $\theta^Tx\ge 0$, but regression is optimizing numerical distance, not classification correctness.

For example, if $y=+1$ and one model predicts $20$ while another predicts $2$, both classify correctly. Yet regression loss treats $20$ as a much larger error than $2$:

$$
(1-20)^2=361
$$

versus

$$
(1-2)^2=1
$$

So regression is not ideal for classification.

---

## 10. Perceptron hypothesis

The perceptron computes:

$$
z=\theta^Tx
$$

then applies the sign:

$$
\boxed{
 h_\theta(x)=
 \begin{cases}
 +1, & \theta^Tx\ge 0 \\
 -1, & \theta^Tx<0
 \end{cases}}
$$

Sometimes this is written as:

$$
h_\theta(x)=\operatorname{sign}(\theta^Tx)
$$

Example:

$$
x=
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}
-4 \\
1 \\
2
\end{bmatrix}
$$

Then:

$$
z=(-4)(1)+(1)(2)+(2)(3)=-4+2+6=4
$$

Therefore:

$$
h_\theta(x)=+1
$$

---

## 11. Perceptron learning idea

The most important intuition is:

> If the classifier gets an example correct, leave the boundary alone. If it gets an example wrong, move the boundary in a direction that makes that example more likely to be correct.

If an example is misclassified, we update:

$$
\boxed{\theta \leftarrow \theta + \eta y_i x_i}
$$

where $\eta$ is the learning rate.

This single rule works for both positive and negative examples.

---

## 12. Understanding the update rule

### Case A: positive sample incorrectly classified

Suppose:

$$
y_i=+1
$$

Then the update is:

$$
\theta \leftarrow \theta + \eta x_i
$$

So the parameter vector moves toward the positive sample.

### Case B: negative sample incorrectly classified

Now:

$$
y_i=-1
$$

Then:

$$
\theta \leftarrow \theta - \eta x_i
$$

So the boundary moves away from the negative sample.

Thus the rule:

$$
\theta \leftarrow \theta + \eta y_i x_i
$$

automatically handles both cases.

---

## 13. Small numerical example

Initialization:

$$
\theta=[0,0,0]
$$

Input:

$$
x=[1,2,3]
$$

Label:

$$
y=+1
$$

Score:

$$
\theta^Tx=0
$$

The point is on the decision boundary, so we update.

Let learning rate be $\eta=1$.

Then:

$$
\theta_{\text{new}} = [0,0,0] + 1(+1)[1,2,3] = [1,2,3]
$$

So:

$$
\boxed{\theta=[1,2,3]}
$$

Now the score becomes:

$$
[1,2,3]^T[1,2,3]=1+4+9=14
$$

and the sample is now strongly predicted as class $+1$.

---

## 14. Perceptron loss function

A common form of perceptron loss is:

$$
\boxed{L_i(\theta)=\max(0,-y_i\theta^Tx_i)}
$$

If a point is correctly classified, for example:

$$
y_i(\theta^Tx_i)=5
$$

then:

$$
-y_i(\theta^Tx_i)=-5
$$

and

$$
L_i=\max(0,-5)=0
$$

So there is no penalty.

If the point is misclassified, for example:

$$
y_i(\theta^Tx_i)=-4
$$

then:

$$
-y_i(\theta^Tx_i)=4
$$

and

$$
L_i=4
$$

Thus misclassified examples contribute positive loss.

---

## 15. Perceptron pseudocode

This is worth memorizing because it often appears in exams or viva questions.

```text
initialize theta = 0

repeat for several epochs:

    shuffle the training examples

    for each (x_i, y_i):

        score = theta^T x_i

        if y_i * score <= 0:

            theta = theta + learning_rate * y_i * x_i

return theta
```

Prediction:

```text
score = theta^T x

if score >= 0:
    predict +1
else:
    predict -1
```

Your notebook implements the same idea directly rather than using a library perceptron.

---

## 16. What is an epoch?

One complete pass through the training dataset is one epoch.

If there are 100 examples and we process all 100 once, that is:

$$
1\text{ epoch}
$$

If we process all 100 examples 50 times, that is:

$$
50\text{ epochs}
$$

The perceptron may make several updates within a single epoch.

---

## 17. Why shuffle the examples?

Suppose examples always arrive in this order:

$$
A,A,A,A,B,B,B,B
$$

The perceptron update depends on the order in which samples are encountered. Shuffling avoids repeatedly exposing the model to the same ordering and usually leads to more reasonable training behavior.

---

## 18. What does convergence mean?

Suppose after one full epoch, the model makes zero mistakes. This means each training example satisfies:

$$
y_i(\theta^Tx_i)>0
$$

If the data is linearly separable, the perceptron convergence theorem guarantees that the algorithm will eventually find a separating hyperplane after a finite number of updates.

This condition is essential:

$$
\boxed{\text{linearly separable}}
$$

---

## 19. What does linearly separable mean?

Data is linearly separable when one straight line or hyperplane can perfectly separate the classes.

Example:

```text
OOOOOO

---------------- decision boundary

XXXXXX
```

This is linearly separable.

But a pattern such as:

```text
X O X
O X O
X O X
```

may not be separable by a single line.

---

## 20. Non-linear decision boundary

Suppose classification depends on:

$$
x_1^2 + x_2^2 < 4
$$

Then the class boundary is a circle:

$$
\boxed{x_1^2 + x_2^2 = 4}
$$

This is clearly not a straight line, so a basic perceptron using only $x_1,x_2$ cannot represent it directly.

However, we can transform the features.

Define:

$$
z_1 = x_1^2,
\qquad
z_2 = x_2^2
$$

Then the boundary becomes:

$$
z_1 + z_2 = 4
$$

or

$$
-4 + z_1 + z_2 = 0
$$

This is now linear in the transformed variables $z_1$ and $z_2$.

This is the main idea behind feature mapping.

---

## 21. Feature mapping

We transform the input via a mapping $\phi(x)$:

$$
\phi(x)=
\begin{bmatrix}
1 \\
x_1^2 \\
x_2^2
\end{bmatrix}
$$

Then a model can learn a linear decision boundary in the transformed feature space even when the original problem is non-linear in the original input coordinates.

This is the key link between linear models and more complex decision regions.

---

## Summary

The perceptron is one of the simplest classification algorithms:

- It computes a linear score $\theta^Tx$
- It uses the sign to predict the class
- It updates only when a sample is misclassified
- It learns a separating hyperplane when the data is linearly separable
- It can be extended to non-linear problems through feature mapping

The central idea is:

$$
\boxed{\text{score } \rightarrow \text{sign } \rightarrow \text{decision boundary } \rightarrow \text{class prediction}}
$$

This is the conceptual foundation for many later machine learning models.

---

## Final note

The official lab asks you to finalize your project title and collect a sample dataset, then apply data augmentation. The mathematics here gives you the theoretical base needed to explain how the perceptron separates classes and why feature design matters in real ML problems.

If you want a clean project report, summarize the perceptron algorithm, explain the decision boundary and label encoding, and then describe your dataset and augmentation steps separately.
