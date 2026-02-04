# Chapter 2: Python Data Structures, Data Cleaning, and Exploratory Data Analysis

## Introduction

In the world of machine learning, there is a popular saying: "garbage in, garbage out." Before you train any machine learning model, you need to **clean** and **explore** your data. Clean, well-understood data is the foundation of useful ML. Messy or biased data leads to misleading or useless models. So: spend time on data quality and exploration *before* you train models.

This chapter has two parts. First, we introduce two Python data structures you will use all the time: **lists** and **dictionaries**. Think of them as the building blocks for storing flower names, measurements, and labels. Second, we move on to **Pandas** DataFrames, data cleaning, exploratory data analysis (EDA), and the **ML golden rule** with **train/validation/test** splits and **cross-validation**. We use the **Iris flower dataset** as our main example: simple measurements (sepal and petal length and width in cm) and three species (setosa, versicolor, virginica). We use **Pandas** for tables and **Matplotlib** or **Seaborn** for simple plots. All code runs in Google Colab; no installations required.

## 2.1 Python Data Structures: Building Blocks for Data Analysis

Before we dive into data cleaning and analysis, we need to understand how Python stores and organizes data. Think of data structures as containers, each designed for specific purposes.

### 2.1.1 Lists: Your First Data Container

Lists are the most basic sequential data structure in Python. Imagine you are listing the names of flower species you measured. A list lets you store these names in order:

**What is a list?**
A list is an ordered collection of items enclosed in square brackets `[]`. Items can be of any type: numbers, text (strings), or even other lists. Most importantly for our purposes, lists are ordered (the first item stays first) and mutable (you can change them after creation).

**Example:**
```python
species = ['setosa', 'versicolor', 'virginica']
```

This list contains three species names. Each name occupies a specific position, starting from 0 (Python uses zero-indexing).

**Accessing elements:**
You can access individual elements using their index position:
- `species[0]` returns 'setosa' (the first species)
- `species[-1]` returns 'virginica' (the last species)
- `species[1:3]` returns ['versicolor', 'virginica'] (a slice from position 1 to 3, excluding 3)

**Common operations:**
Lists support various operations you will use often:
- **Appending**: `species.append('new_species')` adds a new item to the end
- **Length**: `len(species)` tells you how many items you have
- **Checking membership**: `'setosa' in species` returns True
- **Sorting**: `species.sort()` arranges items alphabetically

**Real-world application:**
Suppose you are grouping flowers by size. You might have one list for "small petal" species and another for "large petal" species:

```python
small_petal = ['setosa']
large_petal = ['versicolor', 'virginica']
```

You could then combine these for a full species list:
```python
all_species = small_petal + large_petal  # Concatenation
```

**When to use lists:**
- Storing sequences of names, IDs, or time points
- Maintaining order is important (first sample to last sample)
- You need to add/remove elements frequently
- Simple iteration through elements

### 2.1.2 Dictionaries: Associating Keys with Values

While lists use numeric positions, dictionaries use meaningful keys. This makes them perfect for storing information about one flower or one sample, where you look up by name rather than position.

**What is a dictionary?**
A dictionary stores data as key-value pairs enclosed in curly braces `{}`. Think of it like a real dictionary where you look up a word (the key) to find its definition (the value).

**Example:**
```python
flower_sample = {
    'species': 'setosa',
    'sepal_length_cm': 5.1,
    'sepal_width_cm': 3.5,
    'petal_length_cm': 1.4,
    'petal_width_cm': 0.2
}
```

This dictionary stores measurements for one Iris flower. Each measurement is associated with a descriptive key.

**Extracting values using keys:**

You retrieve a value by putting its key in square brackets. The key must match exactly (including spelling and capital letters).

```python
flower_sample['species']           # Returns 'setosa'
flower_sample['sepal_length_cm']   # Returns 5.1
flower_sample['petal_length_cm']    # Returns 1.4
```

If you use a key that does not exist, Python raises a **KeyError** and your code stops. For example, `flower_sample['unknown_key']` would cause an error.

To avoid that, you can use **`.get()`**. If the key exists, you get the value. If it does not, you get `None`, or a default value you choose:

```python
flower_sample.get('species')           # Returns 'setosa'
flower_sample.get('unknown_key')       # Returns None (no error)
flower_sample.get('unknown_key', 0)    # Returns 0 because the key is missing
```

**Adding and modifying key-value pairs:**

Dictionaries are mutable. You can add a new key-value pair by assigning a value to a new key:

```python
flower_sample['petal_color'] = 'white'   # Adds a new key
```

To change an existing value, use the same assignment with a key that already exists:

```python
flower_sample['sepal_length_cm'] = 5.3   # Updates the value
```

**Other key-value manipulations:**

- **Remove a key-value pair:** use `del` or `.pop()`.
  - `del flower_sample['petal_color']` removes that key and its value. If the key is missing, you get a KeyError.
  - `flower_sample.pop('petal_color')` also removes it and returns the value. You can give a default: `flower_sample.pop('petal_color', None)` so that if the key is missing, you get `None` instead of an error.

```python
val = flower_sample.pop('petal_color', None)   # Removes 'petal_color', returns value; if missing, returns None
```

- **Check if a key exists:** use `in`.

```python
'species' in flower_sample      # True
'unknown_key' in flower_sample  # False
```

- **Get all keys, all values, or all key-value pairs:** use `.keys()`, `.values()`, or `.items()`. These are useful when you loop over the dictionary.

```python
flower_sample.keys()    # All keys
flower_sample.values()  # All values
flower_sample.items()   # Pairs (key, value)
```

Example: loop over each key-value pair and print it:

```python
for key, value in flower_sample.items():
    print(key, ":", value)
```

- **Update one dictionary with another:** use `.update()`. Keys from the other dictionary are added or overwritten.

```python
extra = {'source': 'lab_A', 'sepal_length_cm': 5.2}
flower_sample.update(extra)   # Adds 'source', updates 'sepal_length_cm' to 5.2
```

**Nested dictionaries for multiple samples:**

You can create a dictionary of dictionaries, for example one entry per flower:

```python
flower_database = {
    'flower_1': {
        'species': 'setosa',
        'sepal_length_cm': 5.1,
        'petal_length_cm': 1.4
    },
    'flower_2': {
        'species': 'versicolor',
        'sepal_length_cm': 6.0,
        'petal_length_cm': 4.2
    },
    'flower_3': {
        'species': 'virginica',
        'sepal_length_cm': 6.5,
        'petal_length_cm': 5.5
    }
}
```

**Accessing nested values:**

When a value is itself a dictionary, you use the key twice: first for the outer dictionary, then for the inner one.

```python
flower_database['flower_2']['species']           # Returns 'versicolor'
flower_database['flower_3']['sepal_length_cm']   # Returns 6.5
```

**When to use dictionaries:**
- Storing one sample's measurements where you look up by sample ID
- Any scenario where meaningful labels are better than numeric positions
- When you need fast lookup by key (dictionaries are optimized for this)

## 2.2 Pandas DataFrames and Indexing

Data usually comes as tables: rows are samples (e.g. flowers), columns are features or measurements. In Python, we work with these tables using **Pandas DataFrames**. Think of a DataFrame as a spreadsheet: you can select columns, filter rows, and compute summaries.

### 2.2.1 What Is a DataFrame?

A **DataFrame** is a two-dimensional table with labeled rows and columns. Each column can hold a different type of data (numbers, text, True/False). For example: flower species (text), sepal length (numbers), and petal length (numbers).

**Creating a simple DataFrame:**

```python
import pandas as pd

flower_data = pd.DataFrame({
    'species': ['setosa', 'versicolor', 'virginica', 'setosa'],
    'sepal_length_cm': [5.1, 6.0, 6.5, 4.9],
    'petal_length_cm': [1.4, 4.2, 5.5, 1.5]
})

print(flower_data)
```

You get a table with four rows and three columns. Rows have an **index** (0, 1, 2, 3 by default); columns have **names** ('species', 'sepal_length_cm', etc.).

**Loading the Iris dataset:**

We use the **Iris flower dataset**: 150 flowers, 3 species (setosa, versicolor, virginica), and 4 measurements (sepal length, sepal width, petal length, petal width in cm). You can load it directly from scikit-learn (no file upload needed in Colab):

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Optional: use clearer column names
df.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
```

**Alternative: load from a CSV file**

If you prefer a file, you can save the Iris data as **`iris.csv`** and load it:

**File:** [iris.csv](https://github.com/marafathussain/ML_book_easy/blob/main/data/iris.csv) (see the Data README for how to generate it)

```python
df = pd.read_csv('iris.csv')   # same folder as notebook (or Colab's root after upload)
```

Always check shape and columns right after loading:

```python
print(df.shape)   # (number of rows, number of columns)
print(df.columns) # Column names
print(df.dtypes)  # Data type of each column
print(df.head())  # First few rows
```

### 2.2.2 Selecting Columns and Rows (Indexing)

**Columns:**

- Single column: `df['species']` returns a **Series** (one column).
- Multiple columns: `df[['sepal_length_cm', 'petal_length_cm']]` returns a **DataFrame** (two columns).

**Rows by position (`.iloc`):**

Use integer positions, like list indexing:

```python
df.iloc[0]        # First row
df.iloc[0:3]      # First three rows
df.iloc[:, 0:2]   # All rows, first two columns
```

**Rows by labels (`.loc`):**

Use index or column names:

```python
df.loc[0, 'species']           # Row 0, column 'species'
df.loc[0:2, 'sepal_length_cm':'petal_length_cm']  # Rows 0 to 2, columns in range
```

**Filtering rows with conditions (boolean indexing):**

Keep only rows that meet a condition:

```python
# Flowers with sepal length > 5.5 cm
long_sepal = df[df['sepal_length_cm'] > 5.5]

# Only setosa
setosa_only = df[df['species'] == 'setosa']

# Setosa with petal length < 2
setosa_small = df[(df['species'] == 'setosa') & (df['petal_length_cm'] < 2)]

# Virginica flowers
virginica = df[df['species'] == 'virginica']
```

You will use these patterns again and again when cleaning and exploring data.

## 2.3 Data Quality Issues

Real data is messy. Before you analyze or model anything, look for common problems.

| Issue | Example |
|-------|--------|
| **Missing values** | NA in measurements, dropped samples |
| **Outliers** | Measurement errors, mislabeled flowers |
| **Inconsistent units** | cm vs mm, wrong scale |
| **Typos / encoding** | "setosa" vs "Setosa", wrong IDs |
| **Class imbalance** | Many of one species, few of another |

We focus on **missing values** and **scaling** in this chapter; outliers and imbalance come up again in later chapters.

## 2.4 Handling Missing Values

Missing values show up as `NaN` (Not a Number) in Pandas, or sometimes as blanks, "NA", or -999. First, see *where* and *how much* is missing:

```python
print(df.isna().sum())   # Count of missing per column
# Or
print(df.isnull().sum())
```

You have two main options: **remove** or **impute**.

### 2.4.1 Option 1: Remove (Drop)

- **Drop rows** with any missing: `df_clean = df.dropna()`
- **Drop columns** with too many missing (e.g. more than 20%):
  ```python
  threshold = len(df) * 0.2
  df_clean = df.dropna(thresh=len(df) - threshold, axis=1)
  ```

Use deletion when missingness is limited and you can afford to lose those rows or columns.

### 2.4.2 Option 2: Impute (Fill In)

**Numeric columns:** Replace missing with **mean** or **median**. Median is often better when you have outliers or skewed data.

```python
# Median imputation
df['sepal_length_cm'].fillna(df['sepal_length_cm'].median(), inplace=True)

# Mean imputation
df['petal_length_cm'].fillna(df['petal_length_cm'].mean(), inplace=True)
```

**Categorical columns:** Use the **mode** (most frequent category):

```python
most_common = df['species'].mode()[0]
df['species'].fillna(most_common, inplace=True)
```

**Rule of thumb:** Do not impute blindly. Check *where* and *why* values are missing. Document what you did so others can reproduce it.

## 2.5 Basic Statistics

A few simple summaries help you understand your data before modeling.

**Central tendency:** **Mean** (average) and **median** (middle value). Use median when data is skewed or has outliers.

**Spread:** **Standard deviation (std)** and **range** (min, max). Std tells you how much values vary around the mean.

**Pandas functions:**

```python
df['sepal_length_cm'].mean()
df['sepal_length_cm'].median()
df['sepal_length_cm'].std()
df['sepal_length_cm'].min()
df['sepal_length_cm'].max()

# All at once
df.describe()
```

`df.describe()` gives count, mean, std, min, quartiles, and max for numeric columns. For categorical columns, use `df['species'].value_counts()` and `df['species'].nunique()`.

**Correlation:** To see how two numeric variables move together:

```python
df['sepal_length_cm'].corr(df['petal_length_cm'])
```

Values near 1 or -1 mean strong linear relationship; near 0 means weak. A **correlation matrix** and a **heatmap** (e.g. with Seaborn) are useful for all numeric columns. We use these in the EDA workflow below.

**What to check:** Odd value ranges (e.g. negative lengths), strongly skewed distributions, and unexpected categories. Fix or document them before modeling.

## 2.6 Exploratory Data Analysis (EDA) Workflow

**EDA** means understanding your data *before* you build models. A simple workflow: **load → summarize → visualize → decide**.

### 2.6.1 Step 1: Load and Inspect

- Load the data: from scikit-learn (section 2.2.1) or `pd.read_csv('iris.csv')`.
- Check **shape**, **columns**, **dtypes**, and **first/last rows**:
  ```python
  # If you loaded Iris from sklearn:
  # df is already from load_iris() + DataFrame as in 2.2.1
  print(df.shape)
  print(df.columns)
  print(df.dtypes)
  print(df.head())
  print(df.tail())
  ```
- Quick sanity check: Do column names match what you expect? Are numeric columns actually numeric? If not, fix with `pd.to_numeric(..., errors='coerce')` or similar.

### 2.6.2 Step 2: Summarize

- **Missing:** `df.isna().sum()` or `df.isnull().sum()`
- **Numeric summary:** `df.describe()`
- **Categorical:** `df['species'].value_counts()`, `df['species'].nunique()`

Look for strange min/max, too many missing in one column, or unexpected categories.

### 2.6.3 Step 3: Visualize

Simple plots go a long way:

| Plot | Use when |
|------|----------|
| **Histogram** | Distribution of one numeric variable |
| **Boxplot** | Compare groups, spot outliers |
| **Scatter plot** | Relationship between two numeric variables |
| **Correlation heatmap** | Many numeric variables at once |

The figures below were generated from the **Iris** dataset. They show how each plot type helps you understand the data.

**Histogram: distribution of sepal length**

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/histogram_sepal_length.png" alt="Histogram of sepal length" />
  <p class="caption"><strong>Figure 2.1.</strong> Histogram of sepal length (cm) across Iris flowers. The x-axis shows sepal length; the y-axis shows how many flowers fall in each bin. This plot reveals the shape of the distribution: symmetric, skewed, or bimodal. For Iris, sepal length is roughly bell-shaped.</p>
</div>

**Boxplot: sepal length by species**

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/boxplot_sepal_by_species.png" alt="Boxplot of sepal length by species" />
  <p class="caption"><strong>Figure 2.2.</strong> Boxplot of sepal length (cm) grouped by species. The box shows the interquartile range (middle 50% of data); the line inside is the median. Whiskers extend to 1.5 times the IQR; points beyond are outliers. This plot lets you compare sepal length across setosa, versicolor, and virginica and spot outliers in each group.</p>
</div>

**Scatter plot: sepal length vs petal length**

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/scatter_sepal_petal.png" alt="Scatter plot sepal vs petal length" />
  <p class="caption"><strong>Figure 2.3.</strong> Scatter plot of sepal length (x-axis) vs petal length (y-axis). Each point is one flower. A strong linear pattern suggests high correlation; a cloud suggests weak or no linear relationship. Outliers appear as isolated points. Use this to explore whether two measurements move together.</p>
</div>

**Correlation heatmap: all measurements**

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/correlation_heatmap.png" alt="Correlation heatmap of Iris measurements" />
  <p class="caption"><strong>Figure 2.4.</strong> Correlation heatmap of Iris measurements (sepal length, sepal width, petal length, petal width). Each cell shows the Pearson correlation between two columns. Red indicates positive correlation; blue indicates negative correlation; white indicates near-zero correlation. The diagonal is 1. Use this to see which measurements tend to increase or decrease together.</p>
</div>

You can generate these plots in your notebook with code like:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.hist(df['sepal_length_cm'].dropna(), bins=15, edgecolor='black')
plt.xlabel('Sepal Length (cm)'); plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length'); plt.show()

# Boxplot by species
df.boxplot(column='sepal_length_cm', by='species')
plt.ylabel('Sepal Length (cm)'); plt.suptitle(''); plt.show()

# Scatter plot
plt.scatter(df['sepal_length_cm'], df['petal_length_cm'], alpha=0.6)
plt.xlabel('Sepal Length (cm)'); plt.ylabel('Petal Length (cm)')
plt.title('Sepal vs Petal Length'); plt.show()

# Correlation heatmap (numeric columns only)
num_cols = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Iris measurements: correlations'); plt.tight_layout(); plt.show()
```

### 2.6.4 Step 4: Decide and Document

Based on Steps 1 to 3, decide:

- **What to drop?** (e.g. rows/columns with too many missing, non-informative)
- **What to impute?** (which columns, mean vs median vs mode)
- **What to scale?** (which features, and with what method, later)
- **What to fix?** (typos, units, encoding)

**Document your choices** in your notebook (comments or a short "Data cleaning checklist") so others, and future you, can reproduce your steps.

## 2.7 Normalization and Scaling

Different features often live on different scales (e.g. sepal length 4–8 cm, petal length 1–7 cm). Many ML algorithms care about scale: features with larger numbers can dominate. **Normalization** or **scaling** puts features on a comparable scale.

- **MinMax scaling:** Map values to a range, usually [0, 1].  
  Formula: `(value - min) / (max - min)`
- **Standardization (Z-score):** Mean 0, standard deviation 1.  
  Formula: `(value - mean) / std`

**Why it matters:** If one measurement ranges 0 to 10 and another 0 to 100, in distance-based methods (e.g. k-NN, clustering) the larger scale can dominate. Scaling balances them.

Use **StandardScaler** from scikit-learn to standardize numeric features. **Fit the scaler on the training set only**, then apply it to the validation and test sets. That keeps the test set truly unseen.

```python
from sklearn.preprocessing import StandardScaler

# Assume X_train, X_val, X_test are already split
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit on train, transform train
X_val_scaled = scaler.transform(X_val)           # Transform val (do not fit)
X_test_scaled = scaler.transform(X_test)         # Transform test (do not fit)
```

The figure below shows sepal length before and after standardization. The top panel displays the original values (cm); the bottom panel shows the same data after StandardScaler, with mean 0 and standard deviation 1. The shape of the distribution stays the same, but the scale changes so that features are comparable.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/scaling_before_after.png" alt="Before and after StandardScaler" />
  <p class="caption"><strong>Figure 2.5.</strong> Sepal length (cm) before (top) and after (bottom) StandardScaler. After standardization, the data has mean 0 and standard deviation 1, making it suitable for distance-based ML algorithms. The distribution shape is preserved.</p>
</div>

**Log transform:** If a numeric column is strongly right-skewed (e.g. counts), a **log transform** (e.g. `log2(x + 1)`) can help before scaling or modeling. The `+ 1` avoids log(0). For Iris, the measurements are already on a similar scale, so log transform is optional.

## 2.8 The ML Golden Rule and Data Splits

### 2.8.1 The ML Golden Rule

**The model must *never* see the data used to evaluate it.**

If the model has seen evaluation data (during training or when you chose the model or tuned hyperparameters), we cannot trust the reported performance. That is why we **hold out** some data and never use it for fitting or for choosing the model. We use that held-out data **only once**, for a final performance check.

**Golden rule:** Evaluation data stays unseen until the very end. No training on it, no model selection on it, no tuning on it.

### 2.8.2 Train, Validation, and Test (Three Splits)

Do not use only train and test. Use **three** splits:

| Split | Purpose | Model sees it? |
|-------|---------|-----------------|
| **Training** | Fit the model (learn parameters) | Yes |
| **Validation** | Choose model, hyperparameters, early stopping | No (used only to *select*, not to fit) |
| **Test** | Final evaluation only (report performance) | No, and use **once** at the end |

- **Training:** Data the model learns from.
- **Validation:** Data we use to compare models or settings; the model never trains on it. We use it to pick the best model or hyperparameters.
- **Test:** Data we use only once to report how well the chosen model generalizes. Never use it for training or for model selection.

Typical split: **60% train, 20% validation, 20% test** (or 70/15/15).

### 2.8.3 How to Split: Train / Validation / Test

Split **once** and fix **`random_state`** so the split is reproducible. Split *before* any scaling or preprocessing; fit scalers on the training data only, then apply them to validation and test.

```python
from sklearn.model_selection import train_test_split

# First: separate test set (e.g. 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Then: split temp into train and validation (e.g. 75% train, 25% val of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 of 80% = 20% of full data
)
# Result: 60% train, 20% val, 20% test
```

Use `X_train`, `y_train` to fit the model; `X_val`, `y_val` to choose models or hyperparameters; `X_test`, `y_test` only for the final evaluation.

The figure below illustrates this three-way split. The full dataset is divided into 60% for training, 20% for validation, and 20% for testing. The training portion is used to fit the model; the validation portion is used to compare models or tune hyperparameters; the test portion is locked away until the very end and used only once to report final performance.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/train_val_test_example.jpg" alt="Train, validation, and test split" />
  <p class="caption"><strong>Figure 2.6.</strong> Splitting the full dataset into 60% training, 20% validation, and 20% test. The training set is used to fit the model. The validation set is used for model selection and hyperparameter tuning; the model never trains on it. The test set is held out completely and used only once for the final performance report. Keeping these roles separate is essential to avoid overfitting and to get a trustworthy estimate of how well the model generalizes.</p>
</div>

### 2.8.4 Cross-Validation

When data is limited, a single train/validation split can be unstable (the validation score depends on which samples ended up in the validation set). **Cross-validation (CV)** rotates which part is the validation set so that every sample is used for validation once.

- **k-fold CV:** Split the (training) data into *k* folds (e.g. 5). Train on *k*−1 folds, validate on the remaining fold. Repeat *k* times so each fold is the validation set once. Average the *k* validation scores (and optionally report their standard deviation).
- **Use CV for:** Model selection, hyperparameter tuning, and getting a more stable estimate of performance. **Do not** include the test set in CV; keep it completely held out and use it only at the end.
- **Tool:** **`cross_val_score`** or **`KFold`** from **`sklearn.model_selection`**.

```python
from sklearn.model_selection import cross_val_score

# 5-fold CV on training data only (do not use X_test, y_test here)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("CV score: %.3f (+/- %.3f)" % (scores.mean(), scores.std()))
```

The figure below shows how 5-fold cross-validation works. The training data is divided into 5 equal parts (folds). In each of the 5 rounds, one fold serves as the validation set (blue) and the other 4 folds are used for training (orange). Every sample is used for validation exactly once. After 5 rounds, you have 5 validation scores; the average (and optionally the standard deviation) gives you a more stable estimate than a single train/validation split.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter2/cv_example.jpg" alt="5-fold cross-validation" />
  <p class="caption"><strong>Figure 2.7.</strong> Example of k-fold cross-validation with k=5. The training data is split into 5 folds. In each round (Fold 1 to Fold 5), one fold is held out as the validation set while the remaining folds are used for training. After 5 rounds, every sample has been in the validation set once. The 5 validation scores are averaged to give a more reliable performance estimate than a single split. The test set is not part of CV; it remains held out for the final evaluation.</p>
</div>

**Rule:** Run CV only on the training (and optionally validation) data. The test set is used once at the very end to report final performance.

**Important:** Split *before* scaling or other preprocessing that uses global statistics. Fit scalers (and similar) only on the training data, then apply them to validation and test. That keeps the test set truly unseen. See section 2.7 for the StandardScaler workflow.

## 2.9 Summary and Key Takeaways

**Python data structures:**  
**Lists** are ordered sequences (e.g. species names, sample IDs). **Dictionaries** are key-value pairs (e.g. flower ID to measurements). Use lists when order matters; use dictionaries when you look up by name.

**Pandas DataFrames:**  
You now use **DataFrames** for tables and **indexing** (`[]`, `.loc`, `.iloc`, boolean filters) to select columns and rows. These are the core tools for loading and manipulating data.

**Data quality:**  
Data often has missing values, outliers, inconsistent units, and typos. We focused on **missing values** (drop vs impute with mean/median/mode).

**Basic statistics:**  
Use **mean**, **median**, **std**, **describe()**, and **correlation** to summarize and check your data before modeling.

**EDA workflow:**  
**Load → Summarize → Visualize → Decide.** Always document what you drop, impute, or transform.

**Normalization and scaling:**  
After EDA, scale numeric features (e.g. with **StandardScaler**) so that features with different scales do not dominate. Fit the scaler on the training set only, then apply it to validation and test.

**ML golden rule:** The model must never see the data used to evaluate it. Evaluation data stays unseen until the very end.

**Train / validation / test:** Split data into **training**, **validation**, and **test** sets (e.g. 60/20/20). Use training to fit the model, validation to choose models or hyperparameters, and test only once for final performance. Split *before* fitting scalers or other preprocessing.

**Cross-validation:** When data is limited, use k-fold CV on the training (and optionally validation) data for a more stable estimate. Do not include the test set in CV; use it only once at the end.

With clean, explored data and a proper train/validation/test split (and CV when needed), you are ready to build and evaluate your first ML models in the next chapter.

## 2.10 Further Reading

**Pandas:**  
- [Pandas documentation](https://pandas.pydata.org/docs/): DataFrames, indexing, and data manipulation.

**Preprocessing:**  
- [Scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html): Scaling, encoding, and imputation.

**Visualization:**  
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/): Histograms, boxplots, heatmaps.

**Practice datasets:**  
- **Iris dataset**: Load with `sklearn.datasets.load_iris()` (section 2.2.1) or use **`iris.csv`** from the [data folder](https://github.com/marafathussain/ML_book_easy/blob/main/data/). Use it for all EDA and train/test examples in this chapter.  
- [UCI ML Repository](https://archive.ics.uci.edu/ml/): e.g. Iris, Wine, Breast Cancer Wisconsin.

---

**End of Chapter 2**

In the next chapter, we introduce machine learning proper: supervised vs unsupervised learning, classification and regression, and key metrics such as accuracy, AUROC, and RMSE. We will train our first models using the clean data and train/validation/test split (and cross-validation when needed) you have just learned to prepare.
