Yes. The easiest way to prepare for this lab is to see the whole topic as one chain:

\[
\boxed{\text{features }x \rightarrow \text{score } \theta^Tx \rightarrow \text{decision boundary} \rightarrow \text{class}}
\]

Then the perceptron is simply an algorithm for learning the \(\theta\) that makes that classification work.

I also made a complete Lab 03 notebook in the same general coding style as your previous labs, including perceptron from scratch, linear/nonlinear boundaries, lifting, label encoding, plots, and the official image-augmentation task:

**:chatgpt-content-reference{index="5"}[Download Lab3_Classification_Perceptron_Augmentation.ipynb](sandbox:/mnt/data/Lab3_Classification_Perceptron_Augmentation.ipynb)**

Your attached official Lab 03 sheet specifically says to finalize the project title, download a sample ML dataset such as one from Mendeley Data, and apply data augmentation. Lab_03_01Apr2026.docx I handled that separately near the end of the notebook.

---

# 1. What is classification?

In **regression**, the output is a continuous number.

For example:

\[
\text{house price}=4.2\text{ million BDT}
\]

\[
\text{temperature}=31.4^\circ C
\]

In **classification**, the output belongs to a discrete class.

| Input | Classification output |
|---|---|
| Email | Spam / Not Spam |
| Medical image | Disease / Healthy |
| Transaction | Fraud / Legitimate |
| Image | Cat / Dog |
| Student information | Pass / Fail |

For binary classification we often represent the output as either

\[
y\in\{0,1\}
\]

or

\[
y\in\{-1,+1\}
\]

For perceptron, \(-1,+1\) is particularly convenient.

---

# 2. Problem formulation

Suppose every example has two features:

\[
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\]

and target:

\[
y\in\{-1,+1\}
\]

Our training dataset is

\[
\mathcal D=
\{(x^{(1)},y^{(1)}),...,(x^{(m)},y^{(m)})\}
\]

The objective is to learn some function

\[
h_\theta(x)
\]

such that for a new unseen \(x\), we correctly predict its class.

For a linear classifier we first calculate a **score**

\[
z=\theta_0+\theta_1x_1+\theta_2x_2
\]

and classify according to its sign:

\[
h_\theta(x)=
\begin{cases}
+1,&z\geq0\\
-1,&z<0
\end{cases}
\]

So essentially:

\[
\boxed{h_\theta(x)=\operatorname{sign}(\theta^Tx)}
\]

---

# 3. What is a decision boundary?

This is one of the most important ideas for tomorrow.

Suppose

\[
z=\theta_0+\theta_1x_1+\theta_2x_2
\]

The classifier says:

\[
z>0\Rightarrow +1
\]

and

\[
z<0\Rightarrow -1
\]

So what happens exactly between them?

\[
z=0
\]

Therefore the **decision boundary** is

\[
\boxed{\theta_0+\theta_1x_1+\theta_2x_2=0}
\]

For two features, this is a **line**.

For example,

\[
-3+2x_1+x_2=0
\]

or

\[
x_2=3-2x_1
\]

Everything on one side gets one class and everything on the other side gets the other class.

Add More

0.511.522.53−0.5−1−1.50.511.522.53−0.5−1

0,0

Give feedback

The line itself isn't really a class. It is where the classifier is exactly undecided:

\[
\theta^Tx=0
\]

---

# 4. Hyperplane vs decision boundary

They're closely related.

For a \(d\)-dimensional input,

\[
x=(x_1,x_2,\ldots,x_d)
\]

the equation

\[
\theta_0+\theta_1x_1+\cdots+\theta_dx_d=0
\]

defines a **hyperplane**.

Its name depends on the dimension:

| Input dimension | Boundary |
|---:|---|
| 1D | Point |
| 2D | Line |
| 3D | Plane |
| \(d\)D | Hyperplane |

The **decision boundary of a linear classifier is a hyperplane**.

So if your teacher asks:

> What is the distinction between hyperplane and decision boundary?

You can say:

> A hyperplane is a geometric object. In a linear classifier, the hyperplane \(\theta^Tx=0\) acts as the decision boundary separating the classes.

---

# 5. What do the parameters mean?

Consider:

\[
\theta_0+\theta_1x_1+\theta_2x_2=0
\]

The vector

\[
\begin{bmatrix}
\theta_1\\
\theta_2
\end{bmatrix}
\]

is perpendicular to the decision boundary.

It determines the **orientation** of the line.

Meanwhile \(\theta_0\) moves the line away from the origin.

For example:

\[
x_1+x_2=0
\]

passes through the origin.

But

\[
x_1+x_2-5=0
\]

does not.

That \(-5\) is the bias/intercept.

---

# 6. Why do we add a constant feature?

This appeared already in your previous regression labs, but now its geometric meaning becomes more important.

Suppose

\[
z=\theta_0+\theta_1x_1+\theta_2x_2
\]

Define:

\[
x_0=1
\]

Then

\[
x=
\begin{bmatrix}
1\\x_1\\x_2
\end{bmatrix}
\]

and

\[
\theta=
\begin{bmatrix}
\theta_0\\\theta_1\\\theta_2
\end{bmatrix}
\]

Now:

\[
\theta^Tx
=
\theta_0(1)+\theta_1x_1+\theta_2x_2
\]

Thus:

\[
\boxed{z=\theta^Tx}
\]

Instead of having separate \(b\) and \(w\),

\[
w^Tx+b
\]

we can incorporate everything inside one vector.

---

# 7. Why is this called "lifting"?

Suppose originally:

\[
(x_1,x_2)
\]

is a 2D point.

We change it to

\[
(1,x_1,x_2)
\]

which is a 3D point.

For example,

\[
(2,4)
\]

becomes

\[
(1,2,4)
\]

We have **lifted** the data from 2D into 3D.

Notice something interesting: every new point has first coordinate \(1\).

So all lifted points sit on the plane

\[
x_0=1
\]

in 3D.

This lets us rewrite an affine boundary such as

\[
2x_1+x_2-3=0
\]

as

\[
\begin{bmatrix}
-3&2&1
\end{bmatrix}
\begin{bmatrix}
1\\x_1\\x_2
\end{bmatrix}
=0
\]

or simply

\[
\boxed{\theta^Tx=0}
\]

This is likely something your instructor could ask in viva.

---

# 8. Target label encoding

Suppose your original labels are:

\[
\text{No}=0,\qquad\text{Yes}=1
\]

Perceptron commonly converts this into:

\[
0\rightarrow-1
\]

\[
1\rightarrow+1
\]

Why?

Because then this one expression tells us whether a prediction is correct:

\[
\boxed{y(\theta^Tx)}
\]

Suppose true label:

\[
y=+1
\]

and

\[
\theta^Tx=5
\]

Then:

\[
y(\theta^Tx)=+5>0
\]

Correct.

Suppose:

\[
y=-1
\]

and score:

\[
\theta^Tx=-3
\]

Then

\[
(-1)(-3)=3>0
\]

Again correct.

But suppose:

\[
y=+1,\quad \theta^Tx=-4
\]

Then

\[
(+1)(-4)=-4<0
\]

Wrong.

Therefore:

\[
\boxed{y_i(\theta^Tx_i)>0\Rightarrow\text{correct}}
\]

\[
\boxed{y_i(\theta^Tx_i)\leq0\Rightarrow\text{incorrect/boundary}}
\]

This elegant expression is why \(-1,+1\) encoding is useful.

---

# 9. Relationship with linear regression

Linear regression uses:

\[
h_\theta(x)=\theta^Tx
\]

Perceptron also begins with:

\[
z=\theta^Tx
\]

So they look similar.

But their interpretation is very different.

| Linear Regression | Perceptron |
|---|---|
| Predict continuous number | Predict class |
| Output is \(\theta^Tx\) | Output is \(\operatorname{sign}(\theta^Tx)\) |
| Usually MSE loss | Perceptron loss |
| Every error affects parameters | Primarily misclassified examples trigger updates |
| Regression | Classification |

You technically **can** perform:

\[
y\in\{-1,+1\}
\]

and fit linear regression, then classify with

\[
\theta^Tx\geq0
\]

but linear regression's objective is minimizing numerical distance:

\[
(y-\hat y)^2
\]

not classification error.

Consider:

\[
y=+1
\]

One model predicts \(20\), another predicts \(2\).

Both classify the sample correctly.

But regression says:

\[
(1-20)^2=361
\]

versus

\[
(1-2)^2=1
\]

It treats the prediction \(20\) as a huge error even though its **class is perfectly correct**.

That is one reason regression loss is not ideal for classification.

---

# 10. Perceptron hypothesis

The perceptron computes:

\[
z=\theta^Tx
\]

then

\[
\boxed{
h_\theta(x)=
\begin{cases}
+1&\theta^Tx\geq0\\
-1&\theta^Tx<0
\end{cases}}
\]

Sometimes this is simply written:

\[
h_\theta(x)=\operatorname{sign}(\theta^Tx)
\]

Example:

\[
x=
\begin{bmatrix}
1\\2\\3
\end{bmatrix}
\]

and

\[
\theta=
\begin{bmatrix}
-4\\1\\2
\end{bmatrix}
\]

Then:

\[
z=(-4)(1)+(1)(2)+(2)(3)
\]

\[
z=-4+2+6=4
\]

Thus:

\[
h_\theta(x)=+1
\]

---

# 11. Perceptron learning idea

The most important intuition is:

> If the classifier gets an example correct, leave the boundary alone. If it gets an example wrong, move the boundary in a direction that makes that example more likely to be correct.

Suppose:

\[
y_i(\theta^Tx_i)\leq0
\]

Then update:

\[
\boxed{
\theta
\leftarrow
\theta+\eta y_ix_i
}
\]

where \(\eta\) is the learning rate.

---

# 12. Understanding the update rule

This becomes much easier if you examine both cases.

### Case A: positive sample incorrectly classified

Suppose:

\[
y_i=+1
\]

The update becomes:

\[
\theta\leftarrow\theta+\eta x_i
\]

Therefore the parameters move **toward the positive sample**.

### Case B: negative sample incorrectly classified

Now:

\[
y_i=-1
\]

Then:

\[
\theta\leftarrow\theta-\eta x_i
\]

It moves in the opposite direction.

So this single equation:

\[
\theta\leftarrow\theta+\eta y_ix_i
\]

handles both cases automatically.

---

# 13. Small numerical example

Initial:

\[
\theta=[0,0,0]
\]

Input:

\[
x=[1,2,3]
\]

Label:

\[
y=+1
\]

Score:

\[
\theta^Tx=0
\]

The point is on the decision boundary, so we update.

Let

\[
\eta=1
\]

Then:

\[
\theta_{\text{new}}
=
[0,0,0]
+
1(+1)[1,2,3]
\]

Therefore:

\[
\boxed{\theta=[1,2,3]}
\]

Now compute its score again:

\[
[1,2,3]^T[1,2,3]
=
1+4+9
=
14
\]

So it now strongly predicts \(+1\).

---

# 14. Perceptron loss function

A common form is:

\[
\boxed{
L_i(\theta)
=
\max(0,-y_i\theta^Tx_i)
}
\]

Consider a correctly classified point:

\[
y_i(\theta^Tx_i)=5
\]

then

\[
-y_i(\theta^Tx_i)=-5
\]

therefore

\[
L_i=\max(0,-5)=0
\]

No penalty.

If incorrectly classified:

\[
y_i(\theta^Tx_i)=-4
\]

then

\[
-y_i(\theta^Tx_i)=4
\]

thus

\[
L_i=4
\]

So incorrectly classified examples contribute positive loss.

---

# 15. Perceptron pseudocode

This is worth memorizing because there is a very good chance of being asked.

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

Your notebook implements exactly this rather than using a library perceptron.

---

# 16. What is an epoch?

One complete pass over the training dataset is one **epoch**.

If there are 100 examples and we process all 100 once:

\[
1\text{ epoch}
\]

Processing them all 50 times:

\[
50\text{ epochs}
\]

The perceptron may make several updates within one epoch.

---

# 17. Why shuffle the examples?

Suppose examples always arrive in this order:

\[
A,A,A,A,B,B,B,B
\]

Perceptron's updates depend on the order in which samples are encountered.

Shuffling avoids repeatedly exposing it to exactly the same ordering and generally gives more reasonable training behavior.

---

# 18. What does convergence mean?

Suppose we complete one entire epoch and make:

\[
0
\]

mistakes.

That means every training example satisfies:

\[
y_i(\theta^Tx_i)>0
\]

If the data is **linearly separable**, the perceptron convergence theorem tells us that the algorithm will eventually find a separating hyperplane after a finite number of updates.

This condition is important:

\[
\boxed{\text{linearly separable}}
\]

---

# 19. What does linearly separable mean?

Data is linearly separable when one straight line/hyperplane can perfectly separate the classes.

Something like:

```text
OOOOOO

---------------- decision boundary

XXXXXX
```

is linearly separable.

But something like:

```text
X O X
O X O
X O X
```

may not be separable by one line.

---

# 20. Non-linear decision boundary

Imagine classification depends on:

\[
x_1^2+x_2^2<4
\]

Then points inside a circle are positive and points outside are negative.

The decision boundary is:

\[
\boxed{x_1^2+x_2^2=4}
\]

That is obviously not a straight line.

A basic perceptron operating on only

\[
x_1,x_2
\]

cannot directly represent it.

But we can transform the features.

Define:

\[
z_1=x_1^2
\]

\[
z_2=x_2^2
\]

Then:

\[
z_1+z_2=4
\]

or

\[
-4+z_1+z_2=0
\]

This is now **linear in \(z_1,z_2\)**.

That is the important trick.

---

# 21. Feature mapping

We transform:

\[
\phi(x)
=
\begin{bmatrix}
1\\
x_1^2\\
x_2^2
\end{bmatrix}
\]

Then the perceptron learns:

\[
\theta^T\phi(x)=0
\]

Although this equation is linear in the transformed features, it corresponds to:

\[
\theta_0+
\theta_1x_1^2+
\theta_2x_2^2=0
\]

in the original space.

That is a **non-linear decision boundary**.

This distinction is extremely important:

> The classifier may be linear in feature space while producing a nonlinear boundary in the original input space.

The nonlinear example and plot are included in your notebook.

---

# 22. Constant lifting vs nonlinear feature mapping

Do not confuse these.

| Transformation | Purpose |
|---|---|
| \((x_1,x_2)\rightarrow(1,x_1,x_2)\) | Incorporate bias/intercept |
| \((x_1,x_2)\rightarrow(1,x_1^2,x_2^2)\) | Enable a nonlinear boundary |
| \((x_1,x_2)\rightarrow(1,x_1,x_2,x_1x_2)\) | Represent interaction/nonlinear relationships |

The first one doesn't magically make nonlinear classes separable.

It mainly provides a convenient representation of the intercept.

---

# 23. Perceptron limitations

The biggest limitation is **non-linearly-separable data**.

If classes overlap, the perceptron may continually update:

\[
\theta^{(1)}
\rightarrow
\theta^{(2)}
\rightarrow
\theta^{(3)}
\rightarrow\cdots
\]

without finding a perfect solution.

It also doesn't naturally give probabilities.

You get something like:

\[
+1
\]

rather than:

\[
P(y=1|x)=0.83
\]

Other important weaknesses are sensitivity to feature scale and sample order, no unique optimum when many separating boundaries exist, and the basic algorithm only being binary.

---

# 24. Perceptron vs gradient descent from your previous labs

This distinction may come up.

For linear regression:

\[
\theta
\leftarrow
\theta-\alpha\nabla J(\theta)
\]

and normally all training samples contribute to the batch gradient.

For the basic perceptron:

\[
\theta
\leftarrow
\theta+\eta y_ix_i
\]

only when sample \(i\) is misclassified.

So the learning mechanism is different.

---

# 25. The official Lab 03 augmentation task

For the PDF task, I selected the **Recyclable Waste Image Dataset** from Mendeley Data as a sensible example. Its public release contains 1,000 author-collected images intended for computer-vision work; the underlying categories include paper, cardboard, plastic, metal, and glass. [Mendeley Data](https://data.mendeley.com/datasets/h5pxbsdz4m/1?utm_source=chatgpt.com)

[Mendeley dataset page](https://data.mendeley.com/datasets/h5pxbsdz4m/1?utm_source=chatgpt.com)

After downloading and extracting it, change this line in the notebook:

```python
DATASET_ROOT = Path(
    "recyclable_waste_dataset"
)
```

to the actual folder location.

The notebook then automatically finds images recursively and generates variants using horizontal flipping, \(+15^\circ\)/\(-15^\circ\) rotations, brightness changes, contrast adjustment, center zoom, and mild Gaussian blur.

The basic idea is:

\[
\text{Original image}
\rightarrow
\begin{cases}
\text{rotated}\\
\text{flipped}\\
\text{brighter}\\
\text{darker}\\
\text{zoomed}\\
\text{etc.}
\end{cases}
\]

The class label stays unchanged.

For example:

\[
\text{plastic bottle}
\xrightarrow{\text{rotation}}
\text{plastic bottle}
\]

This artificially increases diversity in the training data.

A major rule is that an augmentation should remain **label preserving**. You shouldn't blindly apply a transformation just because a library provides it.

Also, in an actual ML experiment, split your original data first and augment only the **training set**. Otherwise related augmented versions can leak into validation/test data and make evaluation unrealistically optimistic.

---

# 26. Viva / quiz Q&A

Here are the questions I'd prepare most carefully.

| Question | Answer |
|---|---|
| **1. What is classification?** | Predicting a discrete class/category from input features. |
| **2. Classification vs regression?** | Regression predicts continuous numerical values; classification predicts discrete labels. |
| **3. What is a hypothesis?** | The learned mapping \(h_\theta(x)\) used to predict output from input. |
| **4. Perceptron hypothesis?** | \(h_\theta(x)=\operatorname{sign}(\theta^Tx)\). |
| **5. What is a decision boundary?** | Set of points where the classifier changes class; for perceptron, \(\theta^Tx=0\). |
| **6. What is a hyperplane?** | A \(d-1\) dimensional linear boundary in a \(d\)-dimensional feature space. |
| **7. What is a hyperplane in 2D?** | A line. |
| **8. What is a hyperplane in 3D?** | A plane. |
| **9. Why use labels \(-1,+1\)?** | Correct classification can conveniently be checked using \(y_i(\theta^Tx_i)>0\). |
| **10. What if labels are 0 and 1?** | Commonly map \(0\to-1\) and \(1\to+1\) for perceptron. |
| **11. What does \(\theta^Tx>0\) mean?** | Point lies on the positive side of the hyperplane. |
| **12. What does \(\theta^Tx<0\) mean?** | Point lies on the negative side. |
| **13. What does \(\theta^Tx=0\) mean?** | Point lies exactly on the decision boundary. |
| **14. Perceptron update rule?** | \(\theta\leftarrow\theta+\eta y_ix_i\) for a misclassified/boundary sample. |
| **15. What is \(\eta\)?** | Learning rate controlling the size of the update. |
| **16. When is an update made?** | Usually when \(y_i(\theta^Tx_i)\leq0\). |
| **17. What if the sample is correctly classified?** | Basic perceptron makes no update. |
| **18. Perceptron loss?** | \(L_i=\max(0,-y_i\theta^Tx_i)\). |
| **19. Why add \(x_0=1\)?** | To absorb the bias/intercept into the feature and parameter vectors. |
| **20. What does lifting mean?** | Mapping data into a higher-dimensional feature representation. |
| **21. Does adding \(x_0=1\) alone solve nonlinear classification?** | No. It incorporates the intercept; nonlinear features are needed for nonlinear boundaries. |
| **22. Why is perceptron related to linear regression?** | Both use a linear score \(\theta^Tx\), but their outputs, objectives and learning rules differ. |
| **23. Why not simply use linear regression for classification?** | Squared-error loss optimizes numeric prediction error rather than class separation. |
| **24. What is linearly separable data?** | Data for which some linear hyperplane perfectly separates all classes. |
| **25. When does perceptron convergence occur?** | The classical convergence guarantee applies when the training data is linearly separable. |
| **26. What happens for nonseparable data?** | The basic perceptron may keep making updates and fail to converge to zero training errors. |
| **27. Can perceptron produce nonlinear boundaries?** | Yes, if the original inputs are transformed into suitable nonlinear features first. |
| **28. Give a nonlinear feature transformation.** | \((x_1,x_2)\to(1,x_1^2,x_2^2)\). |
| **29. What is an epoch?** | One complete pass through all training examples. |
| **30. Why shuffle each epoch?** | Perceptron is order-sensitive, so shuffling prevents repeatedly using the same sequence. |
| **31. Does perceptron output probability?** | No. Basic perceptron provides a class based on the sign of its score. |
| **32. What is data augmentation?** | Creating transformed versions of existing training examples to increase data diversity. |
| **33. Why augment images?** | To improve robustness and reduce overfitting by exposing the model to realistic variations. |
| **34. Should test images be augmented and inserted into training?** | No; that causes information leakage. |
| **35. What makes an augmentation valid?** | It should preserve the ground-truth class/meaning of the original sample. |

---

# 27. Three questions I would especially expect

### "Why is \(y\theta^Tx\) useful?"

Because both correct situations become positive:

\[
(+1)(+score)>0
\]

and

\[
(-1)(-score)>0
\]

while incorrect classifications give a negative value.

Therefore one inequality handles both classes:

\[
\boxed{y_i\theta^Tx_i>0}
\]

---

### "How can a linear classifier produce a nonlinear boundary?"

Not by changing the perceptron itself.

Change the representation:

\[
x\rightarrow\phi(x)
\]

For example:

\[
(x_1,x_2)
\rightarrow
(1,x_1^2,x_2^2)
\]

Then the classifier is linear in \(\phi(x)\):

\[
\theta^T\phi(x)=0
\]

but nonlinear in the original \(x\):

\[
\theta_0+
\theta_1x_1^2+
\theta_2x_2^2=0
\]

---

### "What exactly does adding 1 do?"

It turns

\[
b+w_1x_1+w_2x_2
\]

into

\[
\theta^Tx
\]

by defining

\[
x=
[1,x_1,x_2]^T
\]

and

\[
\theta=
[b,w_1,w_2]^T
\]

So the bias becomes an ordinary parameter.

---

## What I would memorize before the lab

The six formulas that basically cover the entire perceptron portion are:

\[
\boxed{z=\theta^Tx}
\]

\[
\boxed{h_\theta(x)=\operatorname{sign}(z)}
\]

\[
\boxed{\theta^Tx=0\quad\text{decision boundary}}
\]

\[
\boxed{y_i\theta^Tx_i>0\quad\text{correct}}
\]

\[
\boxed{L_i=\max(0,-y_i\theta^Tx_i)}
\]

\[
\boxed{\theta\leftarrow\theta+\eta y_ix_i
\quad\text{if }y_i\theta^Tx_i\leq0}
\]

If you can explain **why** each of these six equations works instead of only memorizing them, you should be in very good shape for the lab.

**Notebook:** Lab3_Classification_Perceptron_Augmentation.ipynb[Lab3_Classification_Perceptron_Augmentation.ipynb](sandbox:/mnt/data/Lab3_Classification_Perceptron_Augmentation.ipynb)
