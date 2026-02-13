# Chapter 4: Beyond Accuracy, Precision, Recall, Imbalance, and Calibration

## Introduction

In Chapter 3, you learned to build and evaluate classifiers using **accuracy** and **AUROC**. But in many real-world problems, especially in biology and medicine, classes are **imbalanced**: one outcome (e.g., a rare disease) is much less frequent than the other. In such cases, accuracy can be misleading: a model that always predicts "no disease" might still have high accuracy if the disease is rare. You need metrics that focus on the **minority class**: Did we find the positives? Did we avoid false alarms?

This chapter covers four topics that are essential for evaluating and improving classifiers when the class distribution is uneven or when you care more about one class than the other:

1. **Precision, recall, and F1-score**, metrics that break down performance into "how many positives did we catch?" and "when we said positive, how often were we right?"
2. **Precision–Recall (PR) curves**, a curve that shows the trade-off between precision and recall as you change the decision threshold, especially useful when positives are rare.
3. **Class weighting vs. oversampling (SMOTE)**, two ways to handle imbalanced data during training: adjust the cost of errors per class, or create synthetic minority samples.
4. **Calibration curves**, how to check whether the model’s predicted probabilities (e.g., "70% chance of virginica") match the actual frequencies in the data.

All examples use the **Iris dataset**. We treat one species as the **positive** class (e.g., virginica) and the others as **negative**, and we sometimes use an imbalanced subset to illustrate why these metrics and methods matter. Code is written for Google Colab; no local installation is required.

---

## 4.1 Feature Selection

**Feature selection** is the process of choosing a subset of relevant features for model training. Removing irrelevant or redundant features can improve model performance, reduce overfitting, and increase interpretability.

**Why select features?**

- **Reduce overfitting**: Fewer features = simpler model = less risk of memorizing noise
- **Improve interpretability**: Easier to understand models with fewer features
- **Faster training**: Less computation with fewer features
- **Remove noise**: Irrelevant features can hurt performance

### 4.1.1 Variance Threshold

**What is variance threshold?**

**Variance threshold** removes features with very low variance (almost constant values). If a feature has the same value for almost all samples, it cannot help distinguish between classes or predict the target.

**The idea:**

If a feature's variance is below a threshold, remove it. Variance measures how spread out the values of that feature are across samples. We estimate it from the training data.

**Estimating variance:**

For a single feature, let $x_1, x_2, \ldots, x_n$ be its values over $n$ samples, and $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ the sample mean. The (sample) variance is:

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
$$

This is the unbiased estimator: we divide by $n-1$ so that on average it equals the true variance. Low $s^2$ means the feature is almost constant; variance threshold removes features whose $s^2$ is below the chosen threshold.

**Example:**

A measurement column has the same value for all 150 flowers (e.g. always 1.0). Variance = 0. This feature provides no information, so remove it.

**How to use variance threshold:**

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < 0.01
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train)
X_val_selected = selector.transform(X_val)

# See which features were removed
selected_features = selector.get_support()
print(f"Kept {selected_features.sum()} features out of {len(selected_features)}")
```

**When to use:**

- As a first step to remove obviously uninformative features
- When you have many features and want a quick filter
- When you suspect some features are constant or nearly constant

### 4.1.2 ANOVA F-test (Classification)

**What is ANOVA F-test?**

**ANOVA F-test** (Analysis of Variance) measures how well a feature separates different classes. It compares the variance between groups (classes) to the variance within groups.

**The F-statistic:**

$$
F = \frac{\text{between-group variance}}{\text{within-group variance}}
$$

**Breaking down:**

- **Between-group variance**: How much the feature's mean differs across classes (larger = better separation)
- **Within-group variance**: How much the feature varies within each class (smaller = more consistent within classes)
- **Larger F**: Feature separates classes better → more useful for prediction

**Example:**

Petal length: mean in virginica = 5.5 cm, in setosa = 1.5 cm → large between-group variance → high F → useful feature.

Sepal width: mean in virginica = 3.0 cm, in setosa = 2.8 cm → small between-group variance → low F → less useful feature.

**How to use ANOVA F-test:**

```python
from sklearn.feature_selection import f_classif, SelectKBest

# Select top k features based on F-statistic
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

# Get feature scores
scores = selector.scores_
feature_names = X_train.columns
top_features = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 features:", top_features)
```

**When to use:**

- For classification tasks
- When you want to rank features by how well they separate classes
- As a filter before training models

### 4.1.3 L1 Regularization (Lasso) for Feature Selection

**Lasso** (covered in section 4.3) automatically performs feature selection by setting some coefficients to exactly zero. Features with zero coefficients are effectively removed.

**How it works:**

- Train a Lasso model
- Check which coefficients are non-zero
- Those features are selected

**Example:**

```python
from sklearn.linear_model import Lasso

# Train Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Find selected features (non-zero coefficients)
selected_features = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
print(f"Lasso selected {len(selected_features)} features")

# Use only selected features
X_train_selected = X_train.iloc[:, selected_features]
X_val_selected = X_val.iloc[:, selected_features]
```

**Advantages of Lasso for feature selection:**

- **Automatic**: No need to specify how many features to keep
- **Model-based**: Selection is tied to prediction performance
- **Interpretable**: Coefficients tell you feature importance

**Comparison of feature selection methods:**

| Method | Type | When to Use |
|--------|------|-------------|
| **Variance Threshold** | Filter | Remove constant/near-constant features |
| **ANOVA F-test** | Filter | Rank features by class separation (classification) |
| **Lasso** | Embedded | Automatic selection during model training |

**Visualization:**

The figure below compares feature selection methods. Variance threshold removes low-variance features. ANOVA F-test ranks features by F-statistic. Lasso sets some coefficients to zero, effectively selecting features.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/feature_selection_comparison.png" alt="Feature selection methods comparison" />
  <p class="caption"><strong>Figure 4.1.</strong> Comparison of feature selection methods. Variance threshold removes features with very low variance (almost constant). ANOVA F-test ranks features by how well they separate classes (higher F-statistic = better). Lasso sets some coefficients to exactly zero, automatically selecting a subset of features. Each method has strengths: variance threshold is fast and simple, ANOVA F-test is interpretable, and Lasso is model-based and automatic.</p>
</div>

## 4.2 Hyperparameter Tuning with GridSearchCV

**Hyperparameters** are settings that control how a model is trained (e.g., tree depth, regularization strength). Unlike model parameters (coefficients), hyperparameters are not learned from data; you must choose them.

**What is GridSearchCV?**

**GridSearchCV** (Grid Search with Cross-Validation) is a method to find the best hyperparameters by:
1. Defining a **grid** of candidate values for each hyperparameter
2. Trying every combination
3. Using **cross-validation** to estimate performance for each combination
4. Selecting the combination with the best CV score

**How GridSearchCV works:**

1. **Define the grid**: List candidate values for each hyperparameter
2. **For each combination**: Train the model and evaluate using k-fold CV
3. **Select best**: Choose the combination with the highest average CV score

**Example:**

Suppose you want to tune a Decision Tree with two hyperparameters:
- **max_depth**: [2, 5, 10]
- **min_samples_leaf**: [1, 5]

*min_samples_leaf*: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left. 

GridSearchCV tries all 6 combinations (3 × 2) and uses 5-fold CV for each:

| max_depth | min_samples_leaf | CV Accuracy (mean) |
|-----------|-----------------|-------------------|
| 2 | 1 | 0.82 |
| 2 | 5 | 0.81 |
| 5 | 1 | 0.88 |
| 5 | 5 | 0.86 |
| 10 | 1 | 0.85 |
| 10 | 5 | 0.83 |

The best combination is max_depth=5, min_samples_leaf=1 with CV accuracy 0.88.

**How to use GridSearchCV:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the parameter grid
param_grid = {
    'max_depth': [2, 5, 10],
    'min_samples_leaf': [1, 5],
    'min_samples_split': [2, 5]
}

# Create the model
tree = DecisionTreeClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(
    tree,                    # Model to tune
    param_grid,              # Parameter grid
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',      # Metric to optimize
    n_jobs=-1                # Use all CPU cores
)

# Fit GridSearchCV (this trains models for all combinations)
grid_search.fit(X_train, y_train)

# Get best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Use the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
```

**Important notes:**

- **Use only training data**: Do not include test data in GridSearchCV
- **Fit on training set**: GridSearchCV uses CV on the training set to select hyperparameters
- **Evaluate on test set**: After selecting best hyperparameters, evaluate the final model on the held-out test set (only once)

**Visualization:**

The figure below illustrates GridSearchCV. The parameter grid defines candidate values. For each combination, k-fold CV is performed (here k=5). The combination with the best average CV score is selected.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/gridsearchcv_concept.png" alt="GridSearchCV concept" />
  <p class="caption"><strong>Figure 4.2.</strong> GridSearchCV workflow. A grid of hyperparameter combinations is defined (e.g., max_depth = [2,5,10], min_samples_leaf = [1,5]). For each combination, k-fold cross-validation (here k=5) is performed on the training data. The combination with the best average CV score is selected. This systematic search helps find hyperparameters that generalize well, not just fit the training data.</p>
</div>

**Tips for using GridSearchCV:**

1. **Start with a coarse grid**: Try a few values first, then refine around the best region
2. **Use appropriate scoring**: Choose a metric that matches your goal (accuracy, AUROC, RMSE, etc.)
3. **Be patient**: GridSearchCV can be slow if the grid is large or models are complex
4. **Consider RandomizedSearchCV**: For very large grids, random search can be faster and often finds good solutions

**RandomizedSearchCV (alternative):**

Instead of trying all combinations, **RandomizedSearchCV** samples a fixed number of random combinations. This is faster for large grids:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions (not fixed lists)
param_distributions = {
    'max_depth': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    tree,
    param_distributions,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    random_state=42
)
random_search.fit(X_train, y_train)
```

## 4.3 Regularization (Ridge and Lasso)

**Regularization** is a technique to prevent overfitting by penalizing large coefficients. Instead of only minimizing prediction error, we minimize error plus a penalty term that encourages smaller coefficients.

**The general idea:**

$$
\text{Loss} = \text{Prediction Error} + \lambda \cdot \text{Penalty}
$$

where $\lambda$ (lambda) controls how much we care about the penalty. Larger $\lambda$ = stronger regularization = simpler models.

### 4.3.1 L2 Regularization (Ridge)

**What is Ridge regression?**

**Ridge regression** adds a penalty proportional to the **sum of squared coefficients**. It shrinks coefficients toward zero but does not set them to exactly zero.

**The equation:**

$$
\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2
$$

**Breaking down the equation:**

- First sum: Usual squared errors (same as linear regression)
- Second sum: Sum of squared coefficients ($\beta_1^2 + \beta_2^2 + \cdots + \beta_p^2$)
- $\lambda$: Regularization strength (hyperparameter to tune)
  - $\lambda = 0$: No regularization (standard linear regression)
  - $\lambda$ large: Strong penalty, coefficients shrink toward zero

**Effect:**

- Coefficients become smaller (shrink toward zero)
- Model becomes smoother and less sensitive to noise
- All features remain in the model (coefficients are not exactly zero)

**When to use Ridge:**

- When you have many features and want to prevent overfitting
- When you believe many features are relevant (do not want to remove any)
- When features are correlated (Ridge handles multicollinearity better than Lasso)

**How to use Ridge:**

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  # alpha is lambda
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_val)
coefficients = ridge.coef_
```

**Choosing $\lambda$ (alpha):** Use cross-validation (e.g., GridSearchCV). Common range: $10^{-4}$ to $10^4$ (log scale). Larger $\lambda$ = simpler model, but may underfit if too large.

### 4.3.2 L1 Regularization (Lasso)

**What is Lasso?**

**Lasso** (Least Absolute Shrinkage and Selection Operator) adds a penalty proportional to the **sum of absolute values of coefficients**. Unlike Ridge, Lasso can set coefficients to exactly zero, effectively performing **feature selection**.

**The equation:**

$$
\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

**Effect:** Some coefficients become exactly zero (features are removed), so Lasso performs automatic feature selection and the model becomes simpler and more interpretable.

**When to use Lasso:** When you want feature selection, have many features and suspect many are irrelevant, or when interpretability is important.

**How to use Lasso:**

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
coefficients = lasso.coef_
selected_features = [i for i, coef in enumerate(coefficients) if coef != 0]
```

**Ridge vs Lasso:**

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|------------|------------|
| **Penalty** | Sum of squared coefficients | Sum of absolute coefficients |
| **Effect on coefficients** | Shrinks toward zero | Can set to exactly zero |
| **Feature selection** | No (all features kept) | Yes (removes irrelevant features) |

### 4.3.3 The unit ball concept: constraint form vs penalty form

Confusion often comes from mixing two different but equivalent ways of thinking about regularization and expecting the boundary to literally appear in the equation. The two views are equivalent; keeping them separate removes the confusion.

**Two formulations, one idea**

There are two mathematically equivalent ways to do regularization:

**1. Constraint form (explicit boundary)**

$$
\min \;\; \text{Loss} \quad \text{subject to} \quad \|\boldsymbol{\beta}\| \leq c
$$

Here the boundary is explicit: anything with norm greater than $c$ is not allowed. This is where the "L1 ball" or "L2 ball" language comes from. For L2, $\|\boldsymbol{\beta}\|$ is the usual Euclidean norm; for L1, it is the sum of absolute values. The allowed set is a ball (circle in 2D for L2, diamond in 2D for L1) of radius $c$.

**2. Penalty form (what we use in practice)**

$$
\min \;\; \text{Loss} + \lambda \|\boldsymbol{\beta}\|
$$

Here there is no explicit boundary. Nothing is forbidden or discarded; all points are allowed. Instead of "you are not allowed past this wall," we say "you can go there, but it will cost you dearly." Large norms are not illegal, they are just too expensive to be optimal. So the boundary did not disappear, it became **implicit** in the trade-off.

**Why values greater than 1 are not a problem**

In the penalty form, the total cost is $\text{Cost} = \text{Loss} + \lambda \|\boldsymbol{\beta}\|$. If one point has very low loss but very large norm, its total cost is large because of the penalty. Compare that to a point with slightly higher loss but much smaller norm: the second point wins. So optimization naturally avoids large norms without ever banning them. Nothing special happens at 1; there is no magic cutoff. Coefficients can be greater than 1 in the penalty form.

**Where the "ball of radius 1" comes from**

For every value of $\lambda$, there exists a value of $c$ such that the constraint formulation and the penalty formulation give the **same solution**. So penalty strength $\lambda$ corresponds to ball size $c$: small $\lambda$ corresponds to a big ball, large $\lambda$ to a small ball. When textbooks draw $\|\boldsymbol{\beta}\| \leq 1$, they are fixing $c = 1$ and letting $\lambda$ absorb the scale. It is pure convenience for drawing, not a requirement that coefficients stay inside 1.

**Mental picture**

Think of two landscapes added together: a **Loss** landscape (lowest at the best-fit solution) and a **penalty** landscape (lowest at zero). We minimize the sum. The minimum ends up somewhere in between. No walls, no discarding, just trade-offs. The "boundary" only appears when we redraw the same problem in constraint form.

**Why the L1 vs L2 geometry still matters**

Even though the penalty form has no explicit boundary, the **shape** of the penalty still determines whether coefficients hit zero. The L2 penalty uses $\beta_j^2$: near zero, this curve is flat (its derivative goes to zero), so the optimizer tends to shrink $\beta_j$ slowly and rarely lands exactly at zero. The L1 penalty uses $|\beta_j|$: it has a sharp corner at zero (a "kink"), so the optimizer can hit zero and stay there. The figure below illustrates this: as a coefficient moves toward zero, the squared term (L2) tapers off slowly, while the absolute term (L1) reaches zero quickly and can push the coefficient exactly to zero.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/l1_l2_penalty_near_zero.png" alt="L1 and L2 penalty shape near zero" />
  <p class="caption"><strong>Figure 4.3a.</strong> L2 penalty (squared, $\beta^2$) is flat near zero, so coefficients shrink slowly and rarely hit exactly zero. L1 penalty (absolute value, $|\beta|$) has a sharp corner at zero, so coefficients can reach zero and stay there. This is why Lasso zeros out weights and Ridge does not.</p>
</div>

**Why does Lasso set many weights to zero but Ridge does not? (geometry)**

In the equivalent constraint picture: Ridge constrains the sum of squared coefficients (an L2 ball, a circle in 2D); Lasso constrains the sum of absolute coefficients (an L1 ball, a diamond in 2D). When we shrink the allowed region, the optimal coefficients sit where the loss contour first touches the constraint. The circle is smooth, so the solution usually stays away from the axes and coefficients rarely hit exactly zero. The diamond has **corners on the axes**, so the solution often hits a corner, meaning one or more coefficients become exactly zero. So Lasso performs feature selection and Ridge does not.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/l1_l2_ball.png" alt="L1 and L2 balls in 2D" />
  <p class="caption"><strong>Figure 4.3b.</strong> L1 ball (diamond) and L2 ball (circle) in 2D coefficient space. The L1 constraint has corners on the axes, so the solution often hits a corner (e.g., $\beta_1 = 0$). The L2 constraint is smooth, so the solution rarely lands exactly on an axis.</p>
</div>

**One-sentence takeaway**

In the penalty form, nothing is discarded and norms can be greater than 1; the "boundary" is an equivalent geometric picture where the effect of $\lambda$ is re-expressed as a constraint radius. The geometry is a visualization tool for the same trade-off, not a literal step-by-step process.

**Elastic Net** combines L1 and L2: $\text{Loss} = \sum (y_i - \hat{y}_i)^2 + \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2$. It balances feature selection (from Lasso) and handling correlated features (from Ridge).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/ridge_vs_lasso.png" alt="Ridge vs Lasso regularization" />
  <p class="caption"><strong>Figure 4.4.</strong> Effect of Ridge (L2) and Lasso (L1) regularization on coefficients. Ridge (left) shrinks all coefficients toward zero but keeps all features. Lasso (right) sets some coefficients to exactly zero (sparse solution), effectively performing feature selection.</p>
</div>

---

## 4.4 The Confusion Matrix: Counting Right and Wrong Predictions

Before we define precision and recall, we need a clear way to count different types of correct and incorrect predictions. The **confusion matrix** does exactly that.

**Binary classification setup:** We have two classes. By convention, the class we care about most (e.g., a disease, or the species "virginica") is called the **positive** class; the other is the **negative** class. The model predicts either "positive" or "negative" for each sample.

**Four possible outcomes:**

| Outcome | Meaning |
|--------|--------|
| **True Positive (TP)** | True label = positive, Prediction = positive ✓ |
| **True Negative (TN)** | True label = negative, Prediction = negative ✓ |
| **False Positive (FP)** | True label = negative, Prediction = positive ✗ (false alarm) |
| **False Negative (FN)** | True label = positive, Prediction = negative ✗ (missed positive) |

A **confusion matrix** is a 2×2 table that counts how many times each outcome occurred:

|  | **Predicted Negative** | **Predicted Positive** |
|--|-------------------------|-------------------------|
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

**Iris example:** Suppose we predict **virginica** (positive) vs **not virginica** (setosa + versicolor = negative). Out of 150 flowers, 50 are virginica and 100 are not. After training a classifier and predicting on a test set, we might get: TN = 45, FP = 5, FN = 4, TP = 6 (if the test set is small). So we correctly identified 6 virginica flowers (TP), missed 4 (FN), wrongly called 5 non-virginica as virginica (FP), and correctly rejected 45 non-virginica (TN).

The figure below shows an example confusion matrix for an Iris classifier (virginica vs non-virginica).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/confusion_matrix_iris.png" alt="Confusion matrix for Iris virginica vs rest" />
  <p class="caption"><strong>Figure 4.5.</strong> Confusion matrix for a binary classifier: virginica (positive) vs setosa + versicolor (negative). Rows = true labels; columns = predictions. Diagonal cells are correct (TN, TP); off-diagonal are errors (FP, FN).</p>
</div>

From these four numbers we can define accuracy, precision, recall, and F1.

---

## 4.5 Precision, Recall, and F1-Score

### 4.5.1 Why accuracy can be misleading

**Accuracy** is the fraction of all predictions that are correct:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

When the positive class is rare, a model that always predicts "negative" can still have high accuracy. For example: 10% of flowers are virginica, 90% are not. If the model always says "not virginica," accuracy = 90%, but we never detect any virginica. So we need metrics that focus on the **positive** class.

### 4.5.2 Recall (Sensitivity, True Positive Rate)

**Recall** answers: *Of all the actual positives, how many did we correctly predict as positive?*

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{TP}{\text{all actual positives}}
$$

- **High recall** means we miss few positives (low FN). Important when missing a positive is costly (e.g., missing a disease).
- **Low recall** means we miss many positives (high FN).

**Iris example:** If there are 10 virginica flowers in the test set and we correctly predict 8 of them as virginica, then TP = 8, FN = 2, and Recall = 8/(8+2) = 0.8 (80%).

### 4.5.3 Precision (Positive Predictive Value)

**Precision** answers: *Of all the samples we predicted as positive, how many were actually positive?*

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{TP}{\text{all predicted positives}}
$$

- **High precision** means few false alarms (low FP). Important when wrongly calling something positive is costly.
- **Low precision** means many false positives.

**Iris example:** If we predicted 12 flowers as virginica and 9 of them were truly virginica, then TP = 9, FP = 3, and Precision = 9/12 = 0.75 (75%).

### 4.5.4 The precision–recall trade-off

Often, **increasing recall** (catching more positives) means lowering the decision threshold, so we predict "positive" more often. That can **increase false positives** and thus **lower precision**. So we usually have a trade-off: we can be more cautious (high precision, lower recall) or more aggressive (high recall, lower precision). The choice depends on the application.

### 4.5.5 F1-Score: Combining precision and recall

The **F1-score** is the **harmonic mean** of precision and recall. It combines both into a single number:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \, TP}{2 \, TP + FP + FN}
$$

**Why harmonic mean?** The harmonic mean is smaller than the arithmetic mean when one of the two is small. So F1 is low if either precision or recall is low. That encourages the model to do well on both, rather than sacrificing one for the other.

**When to use which:**
- **Recall** when missing a positive is worse (e.g., screening for disease).
- **Precision** when false positives are worse (e.g., when a positive triggers an expensive follow-up).
- **F1** when you want a single balance between precision and recall (e.g., default for imbalanced classification).

The figure below shows precision, recall, and F1 for an Iris classifier (virginica vs rest) at a fixed threshold, and how they relate to the confusion matrix.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/precision_recall_f1_iris.png" alt="Precision, recall, F1 for Iris classifier" />
  <p class="caption"><strong>Figure 4.6.</strong> Precision, recall, and F1-score for a binary Iris classifier (virginica vs non-virginica). Bars show the values; the confusion matrix counts (TP, FP, FN) that go into each metric are indicated.</p>
</div>

**How to compute in Python:**

```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# y_true: true labels (0 = negative, 1 = positive)
# y_pred: predicted labels
# For binary: positive_label=1 by default
precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)
cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]
```

---

## 4.6 Precision–Recall Curves

So far we have assumed a **fixed** decision threshold (e.g., predict "positive" if predicted probability > 0.5). In practice, you can **vary the threshold**: a lower threshold means we predict "positive" more often (higher recall, usually lower precision); a higher threshold means we are stricter (higher precision, usually lower recall). A **Precision–Recall (PR) curve** shows this trade-off.

**What is a PR curve?** For each possible threshold, we compute precision and recall on the test set. We then plot **Recall** on the x-axis and **Precision** on the y-axis. Each point on the curve corresponds to one threshold. A good classifier has high precision and high recall, so the curve stays near the top-right corner. The **area under the PR curve (AUPRC** or **AP)** summarizes overall performance; higher is better (maximum 1.0).

**Why use PR curves when data are imbalanced?** When the positive class is rare, the ROC curve can look good even for a weak classifier, because the true negative rate (specificity) is high simply because there are so many negatives. The PR curve focuses on positives: both axes depend on TP, FP, FN, so it is more informative when positives are rare.

**Iris example:** We train a classifier for virginica (positive) vs rest. We obtain predicted probabilities for the test set, then sweep the threshold from 0 to 1 and plot precision vs recall. The figure below shows an example PR curve for this setup.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/pr_curve_iris.png" alt="Precision-Recall curve for Iris" />
  <p class="caption"><strong>Figure 4.7.</strong> Precision–Recall curve for an Iris classifier (virginica vs non-virginica). Each point corresponds to a different decision threshold. The area under the curve (AUPRC) summarizes performance; a curve that stays high and to the right is better.</p>
</div>

**How to compute in Python:**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# y_true: binary labels, y_scores: predicted probabilities for the positive class
precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)  # Area under PR curve
# Plot: plt.plot(recall_vals, precision_vals)
```

---

## 4.7 Class Weighting vs. Oversampling (SMOTE)

When one class is much rarer than the other, many classifiers tend to predict the majority class most of the time. Two common ways to address this are **class weighting** and **oversampling** (e.g., **SMOTE**).

### 4.7.1 The problem: imbalanced training data

Suppose we have 20 virginica flowers (positive) and 80 non-virginica (negative). A classifier trained to minimize overall error might learn to predict "negative" almost always, because that minimizes the number of mistakes on the training set. As a result, recall for the positive class is poor. We want the model to pay more attention to the minority class.

### 4.7.2 Class weighting (balanced weights)

**Idea:** Make each **error** on the minority class count more than an error on the majority class. Many algorithms (e.g., logistic regression, SVM, decision trees in scikit-learn) accept a `class_weight` argument. Setting `class_weight='balanced'` sets the weight of each class inversely proportional to how often it appears: the rarer class gets a higher weight. So the loss function penalizes mistakes on virginica more than mistakes on non-virginica.

**Pros:** No change to the number of samples; no synthetic data; easy to use.  
**Cons:** Does not add new information; it only reweights existing examples.

**Formula (conceptually):** If class \(k\) has \(n_k\) samples and there are \(K\) classes, a common choice is \(w_k = n / (K \cdot n_k)\), so that the total weight per class is the same. The classifier then minimizes a weighted sum of errors.

### 4.7.3 Oversampling and SMOTE

**Idea:** Increase the number of **training** samples from the minority class so the classifier sees more positives. **Random oversampling** duplicates existing minority samples. **SMOTE (Synthetic Minority Over-sampling Technique)** goes further: it creates **new** synthetic minority samples in feature space.

**How SMOTE works (simplified):** For each minority example, SMOTE looks at its nearest minority neighbors. It then creates new points along the line segments between the example and one of its neighbors. So we get new synthetic points that lie "between" existing minority points, rather than exact copies. This can broaden the minority region and help the decision boundary.

**Pros:** The model sees more minority examples; can improve recall.  
**Cons:** Risk of overfitting to synthetic or duplicated points; can increase training time; need to use SMOTE only on the training set, not on validation/test.

**Iris example:** We take an imbalanced subset (e.g., 25 virginica, 75 non-virginica). We train two models: one with `class_weight='balanced'` and one on data oversampled with SMOTE so that the training set has roughly equal numbers of positive and negative samples. The figure below illustrates the idea: we compare decision boundaries or performance (e.g., recall) for the two approaches.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/class_weight_vs_smote_iris.png" alt="Class weighting vs SMOTE on Iris" />
  <p class="caption"><strong>Figure 4.8.</strong> Handling imbalanced Iris data (virginica = minority). Left: no balancing. Center: class_weight='balanced'. Right: SMOTE oversampling. The decision boundary and/or recall for virginica can improve with balancing or SMOTE.</p>
</div>

**How to use in Python:**

```python
# Class weighting (e.g. Logistic Regression or SVC)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# SMOTE (apply only to training data)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
model.fit(X_train_resampled, y_train_resampled)
```

---

## 4.8 Calibration Curves

A classifier often outputs a **probability** (e.g., "probability of virginica = 0.7"). We use a threshold (e.g., 0.5) to turn this into a class prediction. But we also care whether these probabilities are **calibrated**: when the model says 70%, do about 70% of such samples actually belong to the positive class?

### 4.8.1 What is calibration?

**Calibrated probabilities** mean: among all samples that received a predicted probability of about \(p\), the fraction that are actually positive should be about \(p\). For example, among flowers with predicted P(virginica) between 0.6 and 0.7, roughly 60–70% should be virginica. If the model says 0.7 but only 40% of those are virginica, the model is **overconfident** (poorly calibrated).

### 4.8.2 Calibration curve (reliability diagram)

A **calibration curve** (or **reliability diagram**) is a plot that helps you check this:

1. **X-axis:** Predicted probability (often binned, e.g., [0–0.1], [0.1–0.2], …).
2. **Y-axis:** **Actual fraction of positives** in each bin (the observed frequency of the positive class in that bin).

If the model is perfectly calibrated, the curve lies along the **diagonal** (predicted = actual). If the curve is **above** the diagonal in a region, the model is **underconfident** there (actual positive rate higher than predicted). If the curve is **below** the diagonal, the model is **overconfident** (actual positive rate lower than predicted).

**Iris example:** We train a probabilistic classifier (e.g., logistic regression) for virginica vs rest. We get predicted probabilities for the test set, bin them, and plot the mean predicted probability vs the fraction of positives in each bin. The figure below shows a typical calibration curve.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter4/calibration_curve_iris.png" alt="Calibration curve for Iris classifier" />
  <p class="caption"><strong>Figure 4.9.</strong> Calibration curve for an Iris classifier (virginica vs rest). Each point is a bin of predicted probability. Y-axis: fraction of samples in that bin that are actually virginica. A well-calibrated model follows the diagonal (dashed).</p>
</div>

### 4.8.3 How to improve calibration

Some models (e.g., logistic regression) are often reasonably calibrated by default. Others (e.g., decision trees, random forests, SVMs) can be over- or underconfident. **Platt scaling** or **isotonic regression** can post-process the predicted probabilities to make them more calibrated; scikit-learn provides **CalibratedClassifierCV** to wrap a base estimator and calibrate its outputs.

**How to compute in Python:**

```python
from sklearn.calibration import calibration_curve

# y_true: binary labels, y_prob: predicted probabilities for positive class
frac_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
# Plot: plt.plot(mean_predicted_value, frac_of_positives)
# Perfect calibration: plt.plot([0, 1], [0, 1], linestyle='--')
```

---

## 4.9 Summary and Key Takeaways

**Feature selection (section 4.1):** Variance threshold, ANOVA F-test, and Lasso can reduce features to improve interpretability and reduce overfitting.

**Hyperparameter tuning (section 4.2):** GridSearchCV (and RandomizedSearchCV) search over parameter grids with cross-validation to find hyperparameters that generalize well.

**Regularization (Ridge and Lasso, section 4.3):** Penalize large coefficients to prevent overfitting. Ridge (L2) shrinks all coefficients; Lasso (L1) can set many to zero (feature selection). Lasso's L1 constraint has corners on the axes, so solutions often hit zero; Ridge's L2 constraint is smooth, so coefficients rarely hit exactly zero.

**Confusion matrix:** Counts TP, TN, FP, FN. Rows = true labels, columns = predictions. Foundation for precision, recall, and F1.

**Precision and recall:**
- **Recall** = TP / (TP + FN): of all actual positives, how many did we find? Use when missing a positive is costly.
- **Precision** = TP / (TP + FP): of all predicted positives, how many were correct? Use when false positives are costly.
- **F1** = harmonic mean of precision and recall; balances both.

**Precision–Recall curve:** Plot precision vs recall as the decision threshold changes. Area under the curve (AUPRC) summarizes performance. Especially useful when the positive class is rare.

**Imbalanced data:**
- **Class weighting:** Give higher loss weight to the minority class (e.g., `class_weight='balanced'`). No new data; easy to use.
- **SMOTE:** Create synthetic minority samples in feature space. Can improve recall; use only on training data; may overfit.

**Calibration:** Check whether predicted probabilities match actual frequencies. Calibration curve: x = predicted probability (binned), y = actual fraction of positives. Well-calibrated model follows the diagonal. Use **CalibratedClassifierCV** or similar if needed.

With these tools, you can evaluate and improve classifiers when accuracy alone is not enough and when one class matters more than the other.

---

## 4.10 Further Reading

**Regularization:**
- [Ridge and Lasso](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
- [Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

**Precision, recall, and calibration:**
- [Precision, recall, F1 in scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures)
- [Precision-Recall curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Imbalanced-learn (SMOTE, etc.)](https://imbalanced-learn.org/stable/)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)

---

## Topics for the Next Class

- **Clustering:** k-means, hierarchical clustering
- **PCA:** variance, components, projections
- **t-SNE and UMAP** for biological data
- **Visualizing biological structure** (e.g., cell populations)

---

**End of Chapter 4**
