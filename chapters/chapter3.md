# Chapter 3: Introduction to Machine Learning

## Introduction

In Chapter 2, you learned to clean and explore data, and to split it into training, validation, and test sets. Now we turn to **machine learning** itself: how do we build models that learn patterns from data and make predictions?

This chapter introduces the core concepts of supervised machine learning. We start by contrasting machine learning with classical statistics, then cover **classification** (predicting categories) and **regression** (predicting numbers). We introduce key metrics to measure how well models perform: **accuracy** and **AUROC** for classification, and **RMSE** for regression. We then explore several fundamental algorithms: Decision Trees, Random Forests, Logistic Regression, Support Vector Machines (SVMs), Linear Regression, and Polynomial Regression. Each algorithm comes with equations that we break down step by step.

We also cover critical concepts that every ML practitioner must understand: **overfitting** and **underfitting**, **regularization** (L1 and L2), **feature selection**, and **hyperparameter tuning** with GridSearchCV. These topics help you build models that generalize well to new data, not just memorize the training examples.

All examples use the **Iris flower dataset**: predicting flower species (setosa, versicolor, virginica) from sepal and petal measurements (length and width in cm). We also use simple regression examples (e.g. petal length vs sepal length) and feature selection on the same measurements. All code runs in Google Colab; no installations required.

## 3.1 Machine Learning vs. Classical Statistics

Before diving into algorithms, it is helpful to understand how machine learning relates to classical statistics. Both use similar mathematical tools, but they ask different questions and prioritize different goals.

**What is classical statistics?**

Classical statistics focuses on **testing hypotheses** and **explaining relationships**. A typical question might be: "Does sepal length differ significantly between setosa and virginica flowers?" The answer comes in the form of a p-value, confidence interval, or effect size. The goal is to understand *why* something happens and *which* variables matter. Interpretability is crucial: you want to know that sepal length is associated with species, and perhaps how strong that association is.

**What is machine learning?**

Machine learning focuses on **prediction**. A typical question might be: "Can we predict which species an Iris flower belongs to based on its sepal and petal measurements?" The answer is a trained model that outputs predictions for new flowers. The goal is accuracy on unseen data. While interpretability can be valuable, it is secondary to prediction performance. The model might use all four measurements; you care less about each individual measurement's contribution and more about whether the overall prediction is correct.

**Key differences:**

| Aspect | Classical Statistics | Machine Learning |
|--------|---------------------|------------------|
| **Main goal** | Explain relationships, test hypotheses | Predict outcomes on new data |
| **Focus** | Which variables matter and why | How well does the model predict |
| **Interpretability** | High priority | Often secondary |
| **Number of variables** | Usually a few (1-10) | Often many (hundreds to thousands) |
| **Output** | p-values, confidence intervals, effect sizes | Predictions, probabilities, scores |
| **Data usage** | Test specific hypotheses | Learn patterns automatically |

**A concrete example:**

Imagine you have the Iris dataset: 150 flowers, 50 of each species (setosa, versicolor, virginica), with sepal and petal length and width.

- **Classical statistics approach:** Test whether sepal length differs between species. You might find that setosa vs virginica has a p-value of 0.001 (significant) and setosa vs versicolor has p-value 0.3 (not significant). You conclude that sepal length is associated with species. You focus on understanding this one relationship.

- **Machine learning approach:** Train a model using all four measurements to predict species. The model learns that sepal length, petal length, and petal width together predict species with high accuracy. You care about the overall accuracy; the individual contributions of each measurement are less important.

**When to use which?**

- Use **classical statistics** when you need to test specific hypotheses, understand causal relationships, or explain which variables matter most.

- Use **machine learning** when you need to make predictions on new data, work with many variables, or prioritize accuracy over interpretability.

**Important note:** These approaches are not mutually exclusive. Many modern analyses combine both: use statistics to identify important variables, then use ML to build predictive models. Both use similar mathematical foundations (e.g., linear models), but the goals and workflows differ.

## 3.2 Classification vs. Regression

Machine learning problems fall into two main categories based on what you want to predict: **classification** and **regression**.

**Classification: predicting categories**

In classification, the target variable is a **category** (also called a **class** or **label**). The model predicts which category a sample belongs to.

**Examples:**
- Predicting Iris species: setosa, versicolor, or virginica (from sepal and petal measurements)
- Predicting tumor type: malignant vs benign
- Predicting drug response: responder vs non-responder

The output is discrete: one of a fixed set of classes. For binary classification (two classes), the model might output probabilities (e.g., 0.7 probability of virginica) or a class label (e.g. virginica or not).

**Regression: predicting numbers**

In regression, the target variable is a **continuous number**. The model predicts a numeric value.

**Examples:**
- Predicting survival time in months
- Predicting drug response level (0 to 100)
- Predicting gene expression level
- Predicting patient age from biomarkers

The output is continuous: any value within a range. The model outputs a number, and you measure error as the difference between predicted and true values.

**Summary table:**

| Task Type | Output | Example Question | Example Answer |
|-----------|--------|------------------|----------------|
| **Classification** | Category (class) | Is this tumor malignant? | "Malignant" or "Benign" |
| **Regression** | Number | What is the survival time? | 24.5 months |

**Why does this distinction matter?**

Different algorithms are designed for different tasks. Decision Trees, Logistic Regression, and SVMs are commonly used for classification. Linear Regression and Polynomial Regression are used for regression. Some algorithms (like Random Forests) can do both, but you must specify the task type when using them.

In the rest of this chapter, we cover algorithms for both classification and regression, along with metrics appropriate for each task type.

## 3.3 Key Metrics: Measuring Model Performance

Before we introduce specific algorithms, we need to know how to measure whether a model is "good." Different metrics are used for classification and regression.

### 3.3.1 Accuracy (Classification)

**What is accuracy?**

Accuracy is the simplest metric for classification: it is the fraction of predictions that are correct.



$$
\text{Accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$



**Example:**

Suppose you have 100 flowers, and your model predicts species for each. If the model predicts correctly for 85 flowers and incorrectly for 15, then:



$$
\text{Accuracy} = \frac{85}{100} = 0.85
$$



**When is accuracy useful?**

Accuracy is intuitive and easy to interpret. It works well when classes are **balanced** (roughly equal numbers in each class). For example, if you have 50 disease and 50 healthy patients, accuracy tells you how often the model is right overall.

**When is accuracy misleading?**

Accuracy can be misleading when classes are **imbalanced**. Imagine a dataset with 95 setosa flowers and 5 virginica flowers. A model that always predicts "setosa" would achieve 95% accuracy, but it is useless because it never identifies virginica. We will cover better metrics for imbalanced data (precision, recall, F1-score) in a later chapter.

**How to compute accuracy:**

```python
from sklearn.metrics import accuracy_score

# y_true: true labels, y_pred: predicted labels
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### 3.3.2 AUROC (Area Under the ROC Curve) for Classification

**What is AUROC?**

AUROC (Area Under the Receiver Operating Characteristic Curve) measures how well a model **ranks** samples. Instead of asking "is the prediction correct?", AUROC asks "do samples with the positive class tend to get higher scores than samples with the negative class?"

**The ROC curve:**

Many classification models output a **score** or **probability** (e.g., probability of "virginica" from 0 to 1). To convert this to a class prediction, you choose a **threshold** (e.g., predict "virginica" if probability > 0.5). Think of the ROC curve as a thought experiment where you slowly relax how picky your classifier is.

First, the axes:
- **TPR (true positive rate)** is the fraction of real positives you correctly catch: TP / (TP + FN).
- **FPR (false positive rate)** is the fraction of real negatives you accidentally flag: FP / (FP + TN).

Now imagine a classifier that outputs a score, and you slide the decision threshold from very strict to completely careless.

At the **very strict** extreme, the threshold is so high that nothing is called positive. No true positives, so TPR = 0. No false positives, so FPR = 0. That is the origin: (0, 0).

As you **lower the threshold**, you start calling more samples positive. Some of those are real positives, so TPR increases. Some are negatives, so FPR increases. This tracing-out of (FPR, TPR) pairs is the ROC curve.

Why do both go to 1 at the far end? At the **most permissive** extreme, the threshold is so low that everything is labeled positive. All actual positives are predicted positive, so TP = (TP + FN) and TPR = 1. All actual negatives are also predicted positive, so FP = (FP + TN) and FPR = 1. So the rightmost point must be (1, 1). It is not a bug or a property of your model, it is a logical consequence of what happens when the threshold says "yes" to everyone.

This reveals something important: the ROC curve always starts at (0, 0) and ends at (1, 1), no matter how good or bad the model is. The shape in between is what matters.

**AUROC** is the area under this curve. It can be interpreted as the probability that a randomly chosen positive example gets a higher score than a randomly chosen negative one. A perfect model bends sharply toward the top-left. A random model meanders along the diagonal (TPR ≈ FPR). A perversely wrong model bows below the diagonal. ROC is threshold-free because it asks: "If I keep changing my mind about what 'positive' means, how well do positives stay ahead of negatives?" That is why it is popular, and also why it can be misleading when class imbalance or absolute error costs matter. The curve going to (1, 1) is not optimism, it is inevitability.

The **AUROC** ranges from 0 to 1:
- **AUROC = 0.5**: Random guessing (no better than flipping a coin)
- **AUROC = 1.0**: Perfect ranking (all positives rank higher than all negatives)
- **AUROC > 0.7**: Generally considered good
- **AUROC > 0.9**: Excellent

**Intuitive explanation:**

Sort all flowers by their model score (e.g. probability of virginica). If flowers that are actually virginica tend to have higher scores than non-virginica flowers, the model has good ranking ability, and AUROC will be high. AUROC does not depend on the specific threshold you choose; it measures ranking quality across all possible thresholds.

**Example:**

Suppose you have 10 flowers with scores and true labels:

| Flower | Score | True Label |
|--------|-------|------------|
| 1 | 0.95 | virginica |
| 2 | 0.88 | virginica |
| 3 | 0.75 | setosa |
| 4 | 0.70 | virginica |
| 5 | 0.65 | setosa |
| 6 | 0.55 | setosa |
| 7 | 0.45 | virginica |
| 8 | 0.30 | setosa |
| 9 | 0.25 | setosa |
| 10 | 0.10 | setosa |

Sorting by score: Flowers 1, 2, 4, 7 (virginica) and 3, 5, 6, 8, 9, 10 (setosa). Most virginica flowers have higher scores than setosa, but flower 7 (virginica) has a lower score than flowers 3, 5, 6 (setosa). This is not perfect ranking, so AUROC would be less than 1.0, but likely above 0.5.

**When to use AUROC:**

- Use AUROC when you care about ranking quality, not just accuracy at one threshold
- Use AUROC when classes are imbalanced (it is more robust than accuracy)
- Use AUROC when you want a metric that does not depend on choosing a specific threshold

**How to compute AUROC:**

```python
from sklearn.metrics import roc_auc_score

# y_true: true binary labels (0 or 1)
# y_scores: predicted scores or probabilities
auroc = roc_auc_score(y_true, y_scores)
print(f"AUROC: {auroc:.3f}")
```

The figure below shows an example ROC curve. The x-axis is False Positive Rate; the y-axis is True Positive Rate. The diagonal dashed line represents random guessing (AUROC = 0.5). A good model's curve rises quickly toward the top-left corner, giving a large area under the curve (high AUROC).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/roc_curve_example.png" alt="ROC curve example" />
  <p class="caption"><strong>Figure 3.1.</strong> Example ROC curve. The diagonal dashed line represents random guessing (AUROC = 0.5). A good model's curve (solid line) rises quickly toward the top-left corner, indicating that the model can achieve high True Positive Rates with low False Positive Rates. The area under the curve (AUROC) quantifies this ranking ability.</p>
</div>

### 3.3.3 RMSE (Root Mean Squared Error) for Regression

**What is RMSE?**

RMSE (Root Mean Squared Error) measures prediction error for regression tasks. It tells you the "typical" error in the same units as your target variable.



$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$



**Breaking down the equation:**

- $y_i$ = true value for sample $i$
- $\hat{y}_i$ = predicted value for sample $i$
- $(y_i - \hat{y}_i)$ = error (difference between true and predicted)
- $(y_i - \hat{y}_i)^2$ = squared error (squaring makes all errors positive and penalizes large errors more)
- $\frac{1}{n}\sum_{i=1}^{n} \ldots$ = mean squared error (MSE), the average of squared errors
- $\sqrt{\ldots}$ = square root, bringing the value back to the original units (e.g., months, expression units)

**Why square the errors?**

Squaring ensures all errors are positive and gives more weight to large errors. If you predict 10 when the true value is 12, the error is 2. If you predict 10 when the true value is 20, the error is 10. Squaring makes the second error (100) much larger than the first (4), which is appropriate because being off by 10 is much worse than being off by 2.

**Why take the square root?**

The square root converts the mean squared error back to the original units. If you are predicting survival in months, RMSE is also in months, making it easier to interpret than MSE (which would be in months squared).

**Example:**

Suppose you predict survival time (in months) for 5 patients:

| Patient | True Survival | Predicted Survival | Error | Squared Error |
|---------|---------------|-------------------|-------|---------------|
| 1 | 24 | 22 | -2 | 4 |
| 2 | 18 | 20 | 2 | 4 |
| 3 | 30 | 28 | -2 | 4 |
| 4 | 12 | 15 | 3 | 9 |
| 5 | 36 | 30 | -6 | 36 |

Mean Squared Error (MSE) = (4 + 4 + 4 + 9 + 36) / 5 = 57 / 5 = 11.4



$$
\text{RMSE} = \sqrt{11.4} \approx 3.38 \text{ months}
$$



Interpretation: On average, predictions are off by about 3.38 months (in a root-mean-square sense).

**When to use RMSE:**

- Use RMSE for regression tasks when errors should be penalized more if they are large
- Use RMSE when you want the error in the same units as your target variable
- Lower RMSE is better (unlike accuracy and AUROC, where higher is better)

**How to compute RMSE:**

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# y_true: true values, y_pred: predicted values
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# Or directly:
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**Summary of metrics:**

| Metric | Task Type | Range | Higher is Better? | Interpretation |
|--------|-----------|-------|-------------------|----------------|
| **Accuracy** | Classification | 0 to 1 | Yes | Fraction of correct predictions |
| **AUROC** | Classification | 0 to 1 | Yes | Ranking quality (0.5 = random, 1.0 = perfect) |
| **RMSE** | Regression | 0 to ∞ | No | Typical error in original units |

## 3.4 Classification Algorithms

We now introduce several classification algorithms, starting with the simplest and building to more complex methods. We use the **Iris dataset** as our running example: predict species (setosa, versicolor, virginica) from sepal and petal length and width. Each algorithm has strengths and weaknesses; understanding when to use each is key to building effective models.

### 3.4.1 Decision Stump

**What is a decision stump?**

A **decision stump** is the simplest possible decision tree: it makes a single yes/no question based on one feature. It splits the data into two groups using a threshold.

**The rule:**



$$
\text{if } x_j \leq t \text{ then predict class } C_1 \text{ else predict class } C_2
$$



where $x_j$ is a feature (e.g., petal length in cm), $t$ is a threshold, and $C_1$ and $C_2$ are the two classes.

**Example:**

Predict species from petal length. The rule might be: "If petal length > 2.5 cm, predict 'virginica or versicolor'; otherwise predict 'setosa'."

**Visualization:**

The figure below shows a decision stump in action. The x-axis is petal length (cm). A vertical line at the threshold divides the space: flowers to the left are predicted as one class (e.g. setosa), flowers to the right as another (e.g. virginica or versicolor).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/decision_stump.png" alt="Decision stump visualization" />
  <p class="caption"><strong>Figure 3.2.</strong> Decision stump for predicting species from petal length. The vertical line at the threshold divides the feature space. Flowers with petal length ≤ threshold are predicted as one class (left region); flowers with petal length > threshold are predicted as another (right region). This is the simplest possible decision rule: one feature, one threshold.</p>
</div>

**When to use decision stumps:**

- As a baseline model (very simple, easy to interpret)
- As building blocks for more complex models (e.g., boosting algorithms)
- When you have strong reason to believe one feature is highly predictive

**Limitations:**

Decision stumps are usually too simple for real-world problems. They can only use one feature and make one split, so they often have low accuracy. More complex models (like full decision trees) usually perform better.

### 3.4.2 Decision Tree

**What is a decision tree?**

A **decision tree** is a sequence of yes/no questions arranged in a tree structure. Starting at the root, you answer questions and follow branches until you reach a leaf node that gives the final prediction.

**How it works:**

1. Start at the root node with a question (e.g., "Is gene A expression > 5?")
2. If yes, go right; if no, go left
3. At the next node, ask another question (e.g., "Is gene B expression > 2?")
4. Continue until you reach a leaf node that gives the final class prediction

**Example tree structure:**

```
Root: Petal length > 2.5?
├─ Yes → Sepal length > 6?
│  ├─ Yes → Predict "virginica"
│  └─ No → Predict "versicolor"
└─ No → Predict "setosa"
```

**Visualization:**

The figure below shows a decision tree diagram. Each internal node (rectangle) asks a question; each branch represents an answer (yes/no or ≤/>); each leaf node (oval) gives a class prediction.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/decision_tree_diagram.png" alt="Decision tree diagram" />
  <p class="caption"><strong>Figure 3.3.</strong> Example decision tree for Iris species prediction. The root node asks "Is petal length > 2.5?" If yes, the tree checks "Is sepal length > 6?" and predicts virginica or versicolor. If petal length is low, it predicts "setosa" directly. Decision trees can use multiple features and make multiple splits, making them more flexible than decision stumps.</p>
</div>

**How trees are built:**

The algorithm chooses splits that best separate the classes. Common criteria include:
- **Gini impurity**: Measures how mixed the classes are in a node (lower is better)
- **Entropy**: Another measure of class mixing
- **Information gain**: How much the split improves class separation

The tree grows until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or no improvement from further splits).

**Advantages:**

- **Interpretable**: You can read the tree like a flowchart
- **Handles non-linear relationships**: Can capture complex patterns
- **No assumptions about data distribution**: Works with any data shape
- **Feature selection**: Automatically identifies which features matter

**Disadvantages:**

- **Overfitting risk**: Deep trees can memorize training data
- **Instability**: Small data changes can lead to very different trees
- **Greedy**: Each split is chosen locally, not globally optimal

**How to use decision trees:**

```python
from sklearn.tree import DecisionTreeClassifier

# Create and train the tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_val)

# Visualize the tree (optional)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=X_train.columns)
plt.show()
```

**Key hyperparameters:**

- **max_depth**: Maximum depth of the tree (deeper = more complex, risk of overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node
- **max_features**: Maximum features to consider for each split

### 3.4.3 Random Forest

**What is a random forest?**

A **random forest** is an **ensemble** method: it combines many decision trees and makes predictions by majority vote (classification) or averaging (regression). Each tree is trained on a random subset of the data and/or a random subset of features.

**The prediction rule (classification):**



$$
\hat{y} = \text{majority vote}\big(\hat{y}_{\text{tree}_1}, \hat{y}_{\text{tree}_2}, \ldots, \hat{y}_{\text{tree}_B}\big)
$$



where $B$ is the number of trees (e.g., 100 or 500), and $\hat{y}_{\text{tree}_i}$ is the prediction from tree $i$.

**Breaking down the equation:**

- Train $B$ decision trees (e.g., $B = 100$)
- Each tree gives a class prediction
- The final prediction is the class that appears most often (majority vote)

**Example:**

Suppose you have 100 trees predicting species for one flower:
- 62 trees predict "virginica"
- 38 trees predict "versicolor"

The majority vote is "virginica," so the random forest predicts "virginica."

**How random forests reduce overfitting:**

1. **Bootstrap sampling**: Each tree is trained on a random sample (with replacement) of the training data. Different trees see different subsets.
2. **Feature randomness**: At each split, each tree considers only a random subset of features (e.g., $\sqrt{p}$ features where $p$ is the total number of features).
3. **Averaging**: Combining many trees smooths out individual tree errors.

**Visualization:**

The figure below illustrates how a random forest works. Many trees are trained on different random subsets of data and features. Each tree makes a prediction, and the final prediction is the majority vote (or average for regression).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/random_forest_concept.png" alt="Random forest concept" />
  <p class="caption"><strong>Figure 3.4.</strong> Random forest concept. Multiple decision trees are trained on random subsets of data and features. Each tree makes a prediction (e.g., setosa, versicolor, or virginica). The final prediction is the majority vote across all trees. This ensemble approach reduces overfitting and improves generalization compared to a single tree.</p>
</div>

**Advantages:**

- **Reduces overfitting**: More robust than a single tree
- **High accuracy**: Often performs very well in practice
- **Handles many features**: Works well with high-dimensional data
- **Feature importance**: Can identify which features matter most across trees

**Disadvantages:**

- **Less interpretable**: Harder to explain than a single tree
- **Slower**: Takes longer to train and predict than a single tree
- **Memory**: Requires storing many trees

**How to use random forests:**

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the forest
forest = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth of each tree
    random_state=42
)
forest.fit(X_train, y_train)

# Make predictions
y_pred = forest.predict(X_val)

# Get feature importances
importances = forest.feature_importances_
```

**Key hyperparameters:**

- **n_estimators**: Number of trees (more is usually better, but slower)
- **max_depth**: Maximum depth of each tree
- **max_features**: Number of features to consider at each split (often $\sqrt{p}$ or $\log_2(p)$)
- **min_samples_split**: Minimum samples to split a node

### 3.4.4 Logistic Regression

**What is logistic regression?**

**Logistic regression** models the **probability** of the positive class using a linear function passed through a "squashing" function (the logistic function) that ensures the output is between 0 and 1.

**The equation:**



$$
P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$



where:
- $P(Y=1 \mid \mathbf{x})$ is the probability that the sample belongs to class 1 (e.g., disease) given features $\mathbf{x}$
- $z$ is a linear combination of features (like linear regression)
- $\beta_0$ is the intercept (bias term)
- $\beta_1, \beta_2, \ldots, \beta_p$ are coefficients (weights) for features $x_1, x_2, \ldots, x_p$
- $e$ is Euler's number (approximately 2.718)
- The function $\frac{1}{1 + e^{-z}}$ is called the **logistic function** or **sigmoid function**

**Breaking down the logistic function:**

The logistic function $\frac{1}{1 + e^{-z}}$ "squashes" any real number $z$ into the range (0, 1):

- When $z$ is large and positive (e.g., $z = 5$): $e^{-5} \approx 0.0067$, so $P \approx \frac{1}{1 + 0.0067} \approx 0.993$ (very high probability)
- When $z = 0$: $e^{0} = 1$, so $P = \frac{1}{1 + 1} = 0.5$ (50% probability)
- When $z$ is large and negative (e.g., $z = -5$): $e^{5} \approx 148$, so $P \approx \frac{1}{1 + 148} \approx 0.007$ (very low probability)

**Example:**

Suppose you have one feature (petal length in cm) and the model learns:

z = -4 + 1.5 \cdot \text{petal\_length}


- If petal length = 2 cm: $z = -4 + 3 = -1$, so $P \approx 0.27$ (27% probability of virginica)
- If petal length = 5 cm: $z = -4 + 7.5 = 3.5$, so $P \approx 0.97$ (97% probability of virginica)

Interpretation: Higher petal length increases the probability of virginica (compared with setosa).

**Visualization:**

The figure below shows the logistic (sigmoid) curve. The x-axis is $z$ (the linear combination); the y-axis is the probability $P(Y=1)$. The curve is S-shaped: it rises slowly for negative $z$, rises quickly around $z = 0$, and levels off near 1 for large positive $z$.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/logistic_curve.png" alt="Logistic regression sigmoid curve" />
  <p class="caption"><strong>Figure 3.5.</strong> The logistic (sigmoid) function. The x-axis is $z$ (linear combination of features); the y-axis is the probability $P(Y=1)$. The curve is S-shaped: when $z$ is very negative, probability is near 0; when $z = 0$, probability is 0.5; when $z$ is very positive, probability is near 1. This function ensures that probabilities always stay between 0 and 1, regardless of the value of $z$.</p>
</div>

**How logistic regression is trained:**

The algorithm finds coefficients $\beta_0, \beta_1, \ldots, \beta_p$ that maximize the **likelihood** of observing the training data. This is typically done using optimization methods (e.g., gradient descent). The loss function is called **log loss** or **cross-entropy loss**.

**Making predictions:**

- **Probability**: Use the logistic function to get $P(Y=1 \mid \mathbf{x})$ (e.g. probability of virginica)
- **Class**: Choose a threshold (usually 0.5). If $P > 0.5$, predict class 1 (e.g. virginica); otherwise predict class 0 (e.g. setosa)

**Advantages:**

- **Interpretable**: Coefficients tell you how each feature affects the probability
- **Probabilistic**: Outputs probabilities, not just class labels
- **Fast**: Quick to train and predict
- **No assumptions about feature distributions**: Works with any feature types (after encoding)

**Disadvantages:**

- **Linear decision boundary**: Assumes a linear relationship between features and log-odds (can be extended with polynomial features)
- **Sensitive to outliers**: Extreme values can influence coefficients
- **Requires feature scaling**: Works better when features are on similar scales

**How to use logistic regression:**

```python
from sklearn.linear_model import LogisticRegression

# Create and train the model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)

# Predict probabilities
y_proba = logreg.predict_proba(X_val)[:, 1]  # Probability of class 1

# Predict classes
y_pred = logreg.predict(X_val)

# Get coefficients
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]
```

**Interpreting coefficients:**

- **Positive coefficient** ($\beta_j > 0$): Increasing feature $x_j$ increases the probability of class 1
- **Negative coefficient** ($\beta_j < 0$): Increasing feature $x_j$ decreases the probability of class 1
- **Larger absolute value**: Stronger effect on the probability

### 3.4.5 Support Vector Machine (SVM)

**What is an SVM?**

A **Support Vector Machine (SVM)** finds a **boundary** (line in 2D, hyperplane in higher dimensions) that separates the two classes. Unlike other classifiers that merely find *any* separating line, SVMs seek the **optimal** one: the boundary that stays **as far as possible** from both classes. This idea is called **maximum-margin classification**.

**Why does the margin matter? An Iris example.**

Consider predicting **Iris setosa** vs **Iris versicolor** using only **petal length** and **petal width**. These two species are linearly separable: we can draw a line to separate them. But many lines could do the job. The figure below shows three possible linear classifiers.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/svm_linear_classifiers.png" alt="Multiple linear classifiers on Iris" />
  <p class="caption"><strong>Figure 3.6a.</strong> Iris setosa vs versicolor (petal length, petal width). Several linear classifiers can separate the classes. Classifier A and B classify all points correctly but run very close to one class, risky for new data. Classifier C misclassifies several points. Which line should we choose?</p>
</div>

- **Classifier A** and **Classifier B** classify the training data correctly, but their boundaries are *too close* to one of the classes. A new flower that falls just across the line could easily be misclassified.
- **Classifier C** misclassifies several points and is clearly poor.

We want a classifier that: (1) separates the classes correctly, and (2) stays **as far away as possible** from both. That is exactly what the SVM does.

**The margin and support vectors:**

The **margin** is the **distance** between the decision boundary and the closest data points from each class. Think of it as the width of a "street" or "corridor" between the two species. The SVM finds the **widest possible street**, the maximum margin.

The data points that lie exactly on the edges of this street (the closest to the boundary) are called **support vectors**. They "support" or define the decision boundary: if you removed all other points, the SVM would still find the same boundary. Only the support vectors matter.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/svm_margin.png" alt="SVM with margin" />
  <p class="caption"><strong>Figure 3.6b.</strong> SVM on Iris setosa vs versicolor. The solid line is the decision boundary. The dashed lines mark the edges of the margin (the "street"). The points on these edges (circled in red) are the support vectors. SVMs maximize the margin to improve generalization on new data.</p>
</div>

**The decision function:**

For a linear SVM, the decision function is:



$$
f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b = \mathbf{w}^\top \mathbf{x} + b
$$



where:
- $\mathbf{w} = (w_1, w_2, \ldots, w_p)$ is a weight vector (determines the *direction* of the boundary)
- $b$ is a bias term (determines the *position* of the boundary, where it is shifted in space)
- $\mathbf{x} = (x_1, x_2, \ldots, x_p)$ is the feature vector

**Making predictions:**

- If $f(\mathbf{x}) > 0$, predict class +1 (e.g., versicolor)
- If $f(\mathbf{x}) < 0$, predict class -1 (e.g., setosa)
- The **decision boundary** is where $f(\mathbf{x}) = 0$

**Breaking down the equation:**

- $\mathbf{w}^\top \mathbf{x}$ is the dot product (sum of element-wise products): $w_1 x_1 + w_2 x_2 + \cdots + w_p x_p$
- Adding $b$ shifts the boundary (moves it left/right, up/down in feature space)
- Both $\mathbf{w}$ and $b$ are **optimized together** during training: the SVM finds the best combination that maximizes the margin while classifying all training points correctly.

**How is the margin maximized? Why does smaller $\|\mathbf{w}\|$ give a larger margin?**

The distance from a point $\mathbf{x}$ to the decision boundary $\mathbf{w}^\top \mathbf{x} + b = 0$ is proportional to $|\mathbf{w}^\top \mathbf{x} + b| / \|\mathbf{w}\|$. In the standard SVM formulation, we scale the model so that the support vectors (the closest points) satisfy $|\mathbf{w}^\top \mathbf{x} + b| = 1$. Therefore, the **margin**, the distance from the boundary to the nearest points, equals $1 / \|\mathbf{w}\|$.

To **maximize** the margin, we must **minimize** $\|\mathbf{w}\|$ (or equivalently $\|\mathbf{w}\|^2$ for easier optimization). Intuitively: if $\mathbf{w}$ is large, the boundary is "steep" and the corridor between the classes is narrow; if $\mathbf{w}$ is small, the boundary is "gentler" and the corridor is wider. The SVM optimization therefore minimizes $\|\mathbf{w}\|^2$ subject to the constraint that all points are correctly classified with a margin of at least 1. The bias $b$ is part of this optimization too, it adjusts the boundary’s position so that the margin is centered between the two classes.

**Kernels for non-linear boundaries:**

When classes cannot be separated by a straight line, SVMs use **kernels** to map features to a higher-dimensional space where a linear boundary exists. Common kernels include:

- **Linear**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^\top \mathbf{x}_j$ (no transformation)
- **Polynomial**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^\top \mathbf{x}_j + r)^d$
- **RBF (Radial Basis Function)**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$

The RBF kernel is very common and can handle complex, curved decision boundaries.

**Advantages:**

- **Effective**: Often achieves high accuracy
- **Memory efficient**: Only support vectors are stored (not all training data)
- **Versatile**: Kernels allow handling non-linear relationships
- **Robust**: Works well with many features

**Disadvantages:**

- **Less interpretable**: Harder to explain than linear models
- **Sensitive to feature scaling**: Features should be standardized
- **Slow on large datasets**: Training time can be long with many samples
- **Hyperparameter tuning**: Kernel choice and parameters (e.g., $C$, $\gamma$) need tuning

**How to use SVMs:**

```python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_val)

# RBF kernel SVM (non-linear)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred = svm_rbf.predict(X_val)
```

**Key hyperparameters:**

- **C**: Controls the trade-off between maximizing margin and minimizing classification errors (larger $C$ = harder margin, less tolerance for misclassification)
- **kernel**: Type of kernel ('linear', 'poly', 'rbf', etc.)
- **gamma**: Kernel coefficient (for RBF and polynomial kernels; larger = more complex boundaries)

## 3.5 Regression Algorithms

We now turn to regression algorithms for predicting continuous numeric values.

### 3.5.1 Linear Regression

**What is linear regression?**

**Linear regression** predicts a target variable $y$ as a linear (straight-line) function of the features. It is one of the simplest and most interpretable regression methods.

**The equation:**



$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$



where:
- $\hat{y}$ is the predicted value
- $\beta_0$ is the **intercept** (predicted $y$ when all features are 0)
- $\beta_1, \beta_2, \ldots, \beta_p$ are **coefficients** (slopes) for features $x_1, x_2, \ldots, x_p$

**Breaking down the equation:**

- **Intercept** ($\beta_0$): The baseline prediction when all features are zero
- **Coefficient** ($\beta_j$): How much $\hat{y}$ changes when $x_j$ increases by 1 unit, holding all other features constant
- **Linear**: The relationship is a straight line (no curves, no interactions unless you add them)

**Example:**

Suppose you predict petal length from sepal length (both in cm):


$$
\hat{y} = -2.5 + 0.9 \cdot \text{sepal\_length}
$$



- If sepal length = 5 cm: $\hat{y} = -2.5 + 4.5 = 2.0$ cm (predicted petal length)
- If sepal length = 6 cm: $\hat{y} = -2.5 + 5.4 = 2.9$ cm
- If sepal length = 7 cm: $\hat{y} = -2.5 + 6.3 = 3.8$ cm

Interpretation: For every 1 cm increase in sepal length, predicted petal length increases by 0.9 cm.

**How linear regression is trained:**

The algorithm finds coefficients $\beta_0, \beta_1, \ldots, \beta_p$ that minimize the **Mean Squared Error (MSE)**:



$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$



This is done using **ordinary least squares (OLS)** or optimization methods (e.g., gradient descent). The solution has a closed form for OLS, making it fast to compute.

**Visualization:**

The figure below shows linear regression in action. Each point is a sample (e.g., one Iris flower: sepal length and some response). The line is the fitted model: $\hat{y} = \beta_0 + \beta_1 x$. The vertical distances from points to the line are the errors (residuals). Linear regression minimizes the sum of squared errors.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/linear_regression.png" alt="Linear regression" />
  <p class="caption"><strong>Figure 3.7.</strong> Linear regression example. Each point represents a sample (e.g., sepal length and petal length for one flower). The solid line is the fitted model $\hat{y} = \beta_0 + \beta_1 x$. The vertical distances from points to the line are the residuals (errors). Linear regression finds the line that minimizes the sum of squared residuals. The relationship is assumed to be linear (straight line).</p>
</div>

**Advantages:**

- **Simple and interpretable**: Easy to understand and explain
- **Fast**: Very quick to train and predict
- **No hyperparameters**: No tuning needed (though you can add regularization)
- **Probabilistic interpretation**: Can provide confidence intervals

**Disadvantages:**

- **Linear assumption**: Assumes a straight-line relationship (may not fit curved data well)
- **Sensitive to outliers**: Extreme values can strongly influence the line
- **Assumes independence**: Features should be independent (or at least not perfectly correlated)

**How to use linear regression:**

```python
from sklearn.linear_model import LinearRegression

# Create and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_val)

# Get coefficients
coefficients = lr.coef_
intercept = lr.intercept_

# Calculate RMSE
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse:.2f}")
```

**Interpreting coefficients:**

- **Positive coefficient** ($\beta_j > 0$): Increasing $x_j$ increases $\hat{y}$
- **Negative coefficient** ($\beta_j < 0$): Increasing $x_j$ decreases $\hat{y}$
- **Larger absolute value**: Stronger effect on $\hat{y}$

### 3.5.2 Polynomial Regression

**What is polynomial regression?**

**Polynomial regression** extends linear regression by adding **powers** of features (e.g., $x^2$, $x^3$) so the model can capture curved relationships.

**The equation (one feature):**



$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_d x^d
$$



where $d$ is the **degree** of the polynomial.

**Breaking down the equation:**

- $d = 1$: Linear (straight line): $\hat{y} = \beta_0 + \beta_1 x$
- $d = 2$: Quadratic (one bend, parabola): $\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2$
- $d = 3$: Cubic (two bends): $\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$
- $d$ large: Very wiggly curves (risk of overfitting)

**Example:**

The relationship between sepal length and petal length might not be perfectly linear. A quadratic model might be:


$$
\hat{y} = -1.0 + 0.5 \cdot \text{sepal\_length} + 0.1 \cdot \text{sepal\_length}^2
$$



This allows the predicted petal length to curve (e.g. increase more slowly at high sepal lengths).

**Visualization:**

The figure below compares linear and polynomial regression. The left panel shows a linear fit (straight line); the right panel shows a polynomial fit (curved line) that better captures the non-linear relationship.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/linear_vs_polynomial.png" alt="Linear vs polynomial regression" />
  <p class="caption"><strong>Figure 3.8.</strong> Comparison of linear (left) and polynomial (right) regression. The linear model assumes a straight-line relationship, which may not fit curved data well. The polynomial model (degree 2) can capture non-linear relationships, such as saturation or U-shaped dose-response curves. However, high-degree polynomials can overfit (memorize noise) if not regularized.</p>
</div>

**Advantages:**

- **Captures non-linear relationships**: Can model curves, saturation, and other complex patterns
- **Flexible**: Can approximate many smooth functions
- **Still interpretable**: Coefficients have meaning (though less intuitive than linear)

**Disadvantages:**

- **Overfitting risk**: High-degree polynomials can memorize training data
- **Extrapolation danger**: Predictions outside the training range can be wildly wrong
- **More parameters**: Requires more data to fit reliably

**How to use polynomial regression:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.transform(X_val)

# Fit linear regression on polynomial features
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Make predictions
y_pred = poly_reg.predict(X_val_poly)

# Or use a pipeline
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
y_pred = poly_pipeline.predict(X_val)
```

**Choosing the degree:**

- Start with $d = 2$ (quadratic)
- Use cross-validation to compare different degrees
- Avoid very high degrees (e.g., $d > 5$) unless you have a lot of data
- Consider regularization (see Chapter 4) to reduce overfitting

## 3.6 Overfitting, Underfitting, and Model Complexity

Understanding **overfitting** and **underfitting** is crucial for building models that generalize well to new data.

**What is underfitting?**

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. The model performs poorly on both training and test data because it cannot learn the relationships.

**Signs of underfitting:**
- High error on training data
- High error on test data
- Model is too simple (e.g., linear model for non-linear data, very shallow tree)

**Example:**

Predicting exam score from hours studied. An underfit model might always predict "70" regardless of study time. It is too simple and ignores the relationship between study time and score.

**What is overfitting?**

**Overfitting** occurs when a model is too complex and memorizes the training data, including noise and random fluctuations. It performs very well on training data but poorly on new (test) data.

**Signs of overfitting:**
- Low error on training data (sometimes near zero)
- High error on test data
- Model is too complex (e.g., very deep tree, high-degree polynomial)

**Example:**

Predicting exam score from hours studied. An overfit model might memorize every student's exact score, creating a wiggly curve that passes through every training point. It fits the training data perfectly but fails on new students.

**What is the right complexity?**

The goal is to find the **sweet spot**: a model that is complex enough to capture real patterns but not so complex that it memorizes noise.

**Visualization:**

The figure below shows the relationship between model complexity and error. As complexity increases, training error decreases (the model fits training data better). However, test error first decreases (the model learns real patterns) but then increases (the model starts memorizing noise). The sweet spot is where test error is minimized.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter3/overfitting_underfitting.png" alt="Overfitting and underfitting" />
  <p class="caption"><strong>Figure 3.9.</strong> The bias-variance trade-off. As model complexity increases, training error decreases (dashed line), but test error (solid line) first decreases then increases. Underfitting occurs when complexity is too low (left): both errors are high. Overfitting occurs when complexity is too high (right): training error is low but test error is high. The sweet spot (middle) balances bias (underfitting) and variance (overfitting).</p>
</div>

**How to measure complexity:**

- **Number of parameters**: More parameters = more complex
- **Tree depth**: Deeper trees = more complex
- **Polynomial degree**: Higher degree = more complex
- **Regularization strength**: Lower regularization = more complex (see Chapter 4)

**Strategies to avoid overfitting:**

1. **Use simpler models**: Start with linear models or shallow trees
2. **Regularization**: Penalize large coefficients (see Chapter 4)
3. **Cross-validation**: Use validation set to choose model complexity
4. **More data**: More training data helps complex models generalize
5. **Early stopping**: Stop training before the model memorizes training data
6. **Feature selection**: Remove irrelevant features (see section 3.7)

**Strategies to avoid underfitting:**

1. **Use more complex models**: Try polynomial features, deeper trees, or non-linear models
2. **Add features**: Include more relevant variables
3. **Reduce regularization**: Allow the model to use larger coefficients
4. **Check for data issues**: Ensure features are informative and data quality is good

**The bias-variance trade-off:**

- **Bias**: Error from overly simplistic assumptions (underfitting)
- **Variance**: Error from sensitivity to small fluctuations in training data (overfitting)
- **Total error** = Bias + Variance + Irreducible error

The goal is to minimize total error by balancing bias and variance.

## 3.7 Further Reading

**Machine learning fundamentals:**
- [Scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html): Comprehensive documentation for all algorithms covered
- [Introduction to Statistical Learning](https://www.statlearning.com/): Free book covering ML concepts with R examples

**Classification:**
- [Logistic Regression explained](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Decision Trees and Random Forests](https://scikit-learn.org/stable/modules/tree.html)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

**Regression:**
- [Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Polynomial Regression](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features)

**Feature selection:**
- [Feature selection in scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)

**Hyperparameter tuning:**
- [GridSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [RandomizedSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

---

**End of Chapter 3**

In the next chapter, we cover feature selection, hyperparameter tuning (GridSearchCV), a summary of Chapter 3, then regularization (Ridge and Lasso), precision, recall, F1-score, handling imbalanced data, and calibration.

