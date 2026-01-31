# Chapter 2: Python Data Structures, Data Cleaning, and Exploratory Data Analysis

## Introduction

In the world of machine learning, there is a popular saying: "garbage in, garbage out." This principle is especially true in biological research, where data quality can make or break your analysis. Before you train any machine learning model, you need to **clean** and **explore** your data. Clean, well-understood data is the foundation of useful ML. Messy or biased data leads to misleading or useless models. In biology, that often means missing samples, typos, batch effects, or wrong units, all of which can lead to wrong conclusions. So: spend time on data quality and exploration *before* you train models.

This chapter has two parts. First, we introduce two Python data structures you will use all the time: **lists** and **dictionaries**. Think of them as the building blocks for storing gene names, sample IDs, expression values, and metadata. Second, we move on to **Pandas** DataFrames, data cleaning, exploratory data analysis (EDA), and the **ML golden rule** with **train/validation/test** splits and **cross-validation**. We use **Pandas** for tables and **Matplotlib** or **Seaborn** for simple plots. All code runs in Google Colab; no installations required.

## 2.1 Python Data Structures: Building Blocks for Biological Data Analysis

Before we dive into data cleaning and analysis, we need to understand how Python stores and organizes data. Think of data structures as containers, each designed for specific purposes, much like how a biologist uses different containers for different specimens.

### 2.1.1 Lists: Your First Data Container

Lists are the most basic sequential data structure in Python. Imagine you are cataloging genes of interest from a recent study. A list lets you store these gene names in order:

**What is a list?**
A list is an ordered collection of items enclosed in square brackets `[]`. Items can be of any type, numbers, text (strings), or even other lists. Most importantly for our purposes, lists are ordered (the first item stays first) and mutable (you can change them after creation).

**Example in biological context:**
```python
genes = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS']
```

This list contains five gene names. Each gene occupies a specific position, starting from 0 (Python uses zero-indexing, a convention inherited from C programming).

**Accessing elements:**
You can access individual elements using their index position:
- `genes[0]` returns 'BRCA1' (the first gene)
- `genes[-1]` returns 'KRAS' (the last gene)
- `genes[1:3]` returns ['TP53', 'EGFR'] (a slice from position 1 to 3, excluding 3)

**Common operations:**
Lists support various operations crucial for biological data manipulation:
- **Appending**: `genes.append('PTEN')` adds a new gene to the end
- **Length**: `len(genes)` tells you how many genes you have
- **Checking membership**: `'TP53' in genes` returns True
- **Sorting**: `genes.sort()` arranges genes alphabetically

**Real-world application:**
Suppose you are analyzing differential gene expression. You might have one list for upregulated genes and another for downregulated genes:

```python
upregulated = ['VEGFA', 'HIF1A', 'EGFR']
downregulated = ['PTEN', 'TP53', 'RB1']
```

You could then combine these for a comprehensive gene list:
```python
all_degs = upregulated + downregulated  # Concatenation
```

**When to use lists:**
- Storing sequences of gene names, sample IDs, or time points
- Maintaining order is important (first sample to last sample)
- You need to add/remove elements frequently
- Simple iteration through elements

### 2.1.2 Dictionaries: Associating Keys with Values

While lists use numeric positions, dictionaries use meaningful keys. This makes them perfect for storing gene metadata where you want to look up information by gene name rather than remembering position numbers.

**What is a dictionary?**
A dictionary stores data as key-value pairs enclosed in curly braces `{}`. Think of it like a real dictionary where you look up a word (the key) to find its definition (the value).

**Example in biological context:**
```python
brca1_info = {
    'gene_name': 'BRCA1',
    'full_name': 'Breast Cancer 1',
    'chromosome': 17,
    'expression': 5.2,
    'p_value': 0.001,
    'significant': True,
    'pathways': ['DNA repair', 'Cell cycle', 'Tumor suppression']
}
```

This dictionary stores comprehensive information about the BRCA1 gene. Each piece of information is associated with a descriptive key.

**Extracting values using keys:**

You retrieve a value by putting its key in square brackets. The key must match exactly (including spelling and capital letters).

```python
brca1_info['gene_name']      # Returns 'BRCA1'
brca1_info['expression']     # Returns 5.2
brca1_info['pathways']       # Returns ['DNA repair', 'Cell cycle', 'Tumor suppression']
```

If you use a key that does not exist, Python raises a **KeyError** and your code stops. For example, `brca1_info['unknown_key']` would cause an error.

To avoid that, you can use **`.get()`**. If the key exists, you get the value. If it does not, you get `None`, or a default value you choose:

```python
brca1_info.get('gene_name')           # Returns 'BRCA1'
brca1_info.get('unknown_key')         # Returns None (no error)
brca1_info.get('unknown_key', 0)      # Returns 0 because the key is missing
```

**Adding and modifying key-value pairs:**

Dictionaries are mutable. You can add a new key-value pair by assigning a value to a new key:

```python
brca1_info['fold_change'] = 2.5    # Adds a new key 'fold_change' with value 2.5
```

To change an existing value, use the same assignment with a key that already exists:

```python
brca1_info['expression'] = 5.8     # Updates 'expression' from 5.2 to 5.8
```

**Other key-value manipulations:**

- **Remove a key-value pair:** use `del` or `.pop()`.
  - `del brca1_info['fold_change']` removes that key and its value. If the key is missing, you get a KeyError.
  - `brca1_info.pop('fold_change')` also removes it and returns the value. You can give a default: `brca1_info.pop('fold_change', None)` so that if the key is missing, you get `None` instead of an error.

```python
val = brca1_info.pop('fold_change', None)   # Removes 'fold_change', returns 2.5; if key missing, returns None
```

- **Check if a key exists:** use `in`.

```python
'gene_name' in brca1_info    # True
'unknown_key' in brca1_info  # False
```

- **Get all keys, all values, or all key-value pairs:** use `.keys()`, `.values()`, or `.items()`. These are useful when you loop over the dictionary.

```python
brca1_info.keys()    # All keys: dict_keys(['gene_name', 'full_name', 'chromosome', ...])
brca1_info.values()  # All values
brca1_info.items()   # Pairs (key, value)
```

Example: loop over each key-value pair and print it:

```python
for key, value in brca1_info.items():
    print(key, ":", value)
```

- **Update one dictionary with another:** use `.update()`. Keys from the other dictionary are added or overwritten.

```python
extra = {'pubmed_id': '12345', 'expression': 6.0}
brca1_info.update(extra)   # Adds 'pubmed_id', updates 'expression' to 6.0
```

**Nested dictionaries for multiple genes:**

For a complete gene database, you can create a dictionary of dictionaries:

```python
gene_database = {
    'BRCA1': {
        'chromosome': 17,
        'type': 'tumor_suppressor',
        'expression': 5.2
    },
    'TP53': {
        'chromosome': 17,
        'type': 'tumor_suppressor',
        'expression': 12.8
    },
    'EGFR': {
        'chromosome': 7,
        'type': 'oncogene',
        'expression': 3.4
    }
}
```

**Accessing nested values:**

When a value is itself a dictionary, you use the key twice (or more): first for the outer dictionary, then for the inner one.

```python
# gene_database['TP53'] is a dictionary, then we take its 'chromosome' key
gene_database['TP53']['chromosome']   # Returns 17
gene_database['EGFR']['expression']   # Returns 3.4
```

**When to use dictionaries:**
- Storing gene annotations where you look up by gene name
- Patient metadata where patient ID is the key
- Any scenario where meaningful labels are better than numeric positions
- When you need fast lookup by key (dictionaries are optimized for this)

## 2.2 Pandas DataFrames and Indexing

Biological data usually comes as tables: rows are samples or genes, columns are features or measurements. In Python, we work with these tables using **Pandas DataFrames**. Think of a DataFrame as a spreadsheet: you can select columns, filter rows, and compute summaries.

### 2.2.1 What Is a DataFrame?

A **DataFrame** is a two-dimensional table with labeled rows and columns. Each column can hold a different type of data (numbers, text, True/False). That matches real biological data: gene names (text), expression (numbers), and significance flags (True/False) often sit side by side.

**Creating a simple DataFrame:**

```python
import pandas as pd

gene_data = pd.DataFrame({
    'Gene': ['BRCA1', 'TP53', 'EGFR', 'MYC'],
    'Expression': [5.2, 12.8, 3.4, 7.1],
    'Chromosome': [17, 17, 7, 8],
    'Significant': [True, True, False, True]
})

print(gene_data)
```

You get a table with four rows (genes) and four columns. Rows have an **index** (0, 1, 2, 3 by default); columns have **names** ('Gene', 'Expression', etc.).

**Loading data from a file:**

In practice, you load data from a CSV or Excel file. A dummy **`gene_expression.csv`** is available in this book's GitHub repo under `ML_book_easy/data/`. Download it and place it in the same folder as your notebook, or use the path from your repo root. In Google Colab, you can load it directly from the raw GitHub URL (replace `YOUR_USERNAME` and `YOUR_REPO` with the actual repo):

```python
# Load a CSV: use the path that matches where you put the file
df = pd.read_csv('gene_expression.csv')   # same folder as notebook

# If using the repo: df = pd.read_csv('ML_book_easy/data/gene_expression.csv')
# Colab from GitHub: df = pd.read_csv('https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/ML_book_easy/data/gene_expression.csv')
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

- Single column: `df['Gene']` returns a **Series** (one column).
- Multiple columns: `df[['Gene', 'Expression']]` returns a **DataFrame** (two columns).

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
df.loc[0, 'Gene']           # Row 0, column 'Gene'
df.loc[0:2, 'Gene':'Expression']  # Rows 0 to 2, columns from Gene to Expression
```

**Filtering rows with conditions (boolean indexing):**

This is especially useful in biology: keep only rows that meet a condition.

```python
# Genes with expression > 5
high_expr = df[df['Expression'] > 5.0]

# Only significant genes
sig_genes = df[df['Significant'] == True]

# Disease samples and age > 50
older_disease = df[(df['Condition'] == 'Disease') & (df['Age'] > 50)]

# Genes on chromosome 17
chr17 = df[df['Chromosome'] == 17]
```

You will use these patterns again and again when cleaning and exploring data.

## 2.3 Data Quality Issues in Biological Data

Real data is messy. Before you analyze or model anything, look for common problems.

| Issue | Example |
|-------|--------|
| **Missing values** | NA in expression, dropped samples |
| **Outliers** | Measurement errors, failed wells |
| **Batch effects** | Different sequencers, dates, or labs |
| **Inconsistent units** | ng/µL vs. nM, FPKM vs. TPM |
| **Typos / encoding** | "male" vs "Male", wrong IDs |
| **Class imbalance** | Many healthy, few disease |

We focus on **missing values** and **scaling** in this chapter; outliers, batch effects, and imbalance come up again in later chapters.

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

**Numeric columns:** Replace missing with **mean** or **median**. Median is often better when you have outliers or skewed data (e.g. gene expression).

```python
# Median imputation (recommended for gene expression)
df['GENE_1'].fillna(df['GENE_1'].median(), inplace=True)

# Mean imputation
df['Age'].fillna(df['Age'].mean(), inplace=True)
```

**Categorical columns:** Use the **mode** (most frequent category):

```python
most_common = df['Tissue'].mode()[0]
df['Tissue'].fillna(most_common, inplace=True)
```

**Rule of thumb:** Do not impute blindly. Check *where* and *why* values are missing (e.g. only in certain conditions or batches). Document what you did so others can reproduce it.

## 2.5 Normalization and Scaling

Different features often live on different scales (e.g. expression 0 to 1000, age 20 to 80). Many ML algorithms care about scale: features with larger numbers can dominate. **Normalization** or **scaling** puts features on a comparable scale.

- **MinMax scaling:** Map values to a range, usually [0, 1].  
  Formula: `(value - min) / (max - min)`
- **Standardization (Z-score):** Mean 0, standard deviation 1.  
  Formula: `(value - mean) / std`

**Why it matters:** Imagine two genes: one ranges 0 to 10, the other 0 to 10,000. In distance-based methods (e.g. k-NN, clustering), the second would overwhelm the first. Scaling balances them.

We use **StandardScaler** and **MinMaxScaler** from scikit-learn in later chapters. For now, it is enough to know that you will typically scale numeric features before training models, and that you should **fit the scaler on the training set only**, then apply it to the test set. We come back to this when we do train/test splits.

**Log transform:** Gene expression and read counts are often right-skewed. A **log transform** (e.g. `log2(x + 1)`) can help before scaling or modeling:

```python
import numpy as np
df['GENE_1_log'] = np.log2(df['GENE_1'] + 1)
```

The `+ 1` avoids log(0).

## 2.6 Basic Statistics for Biological Data

A few simple summaries help you understand your data before modeling.

**Central tendency:** **Mean** (average) and **median** (middle value). Use median when data is skewed or has outliers.

**Spread:** **Standard deviation (std)** and **range** (min, max). Std tells you how much values vary around the mean.

**Pandas functions:**

```python
df['GENE_1'].mean()
df['GENE_1'].median()
df['GENE_1'].std()
df['GENE_1'].min()
df['GENE_1'].max()

# All at once
df.describe()
```

`df.describe()` gives count, mean, std, min, quartiles, and max for numeric columns. For categorical columns, use `df['Condition'].value_counts()` and `df['Condition'].nunique()`.

**Correlation:** To see how two numeric variables move together:

```python
df['GENE_1'].corr(df['GENE_2'])
```

Values near 1 or -1 mean strong linear relationship; near 0 means weak. For many genes, a **correlation matrix** and a **heatmap** (e.g. with Seaborn) are useful. We use these in the EDA workflow below.

**What to check:** Odd value ranges (e.g. negative expression, ages > 150), strongly skewed distributions, and unexpected categories. Fix or document them before modeling.

## 2.7 Exploratory Data Analysis (EDA) Workflow

**EDA** means understanding your data *before* you build models. A simple workflow: **load → summarize → visualize → decide**.

### 2.7.1 Step 1: Load and Inspect

- Load the file: `pd.read_csv()`, `pd.read_excel()`, etc. (e.g. `gene_expression.csv` from the repo's `ML_book_easy/data/` folder).
- Check **shape**, **columns**, **dtypes**, and **first/last rows**:
  ```python
  df = pd.read_csv('gene_expression.csv')  # or path to your file
  print(df.shape)
  print(df.columns)
  print(df.dtypes)
  print(df.head())
  print(df.tail())
  ```
- Quick sanity check: Do column names and sample IDs match what you expect? Are numeric columns actually numeric? If not, fix with `pd.to_numeric(..., errors='coerce')` or similar.

### 2.7.2 Step 2: Summarize

- **Missing:** `df.isna().sum()` or `df.isnull().sum()`
- **Numeric summary:** `df.describe()`
- **Categorical:** `df['Condition'].value_counts()`, `df['Condition'].nunique()`

Look for strange min/max, too many missing in one column, or unexpected categories.

### 2.7.3 Step 3: Visualize

Simple plots go a long way:

| Plot | Use when |
|------|----------|
| **Histogram** | Distribution of one numeric variable |
| **Boxplot** | Compare groups, spot outliers |
| **Scatter plot** | Relationship between two numeric variables |
| **Correlation heatmap** | Many numeric variables at once |

**Example: histogram and boxplot**

```python
import matplotlib.pyplot as plt

# Histogram of gene expression
plt.hist(df['GENE_1'], bins=30, edgecolor='black')
plt.xlabel('Expression')
plt.ylabel('Frequency')
plt.title('Distribution of GENE_1')
plt.show()

# Boxplot by condition
df.boxplot(column='GENE_1', by='Condition')
plt.ylabel('Expression')
plt.title('GENE_1 by Condition')
plt.show()
```

**Example: correlation heatmap (first few genes)**

```python
import seaborn as sns

gene_cols = ['GENE_1', 'GENE_2', 'GENE_3', 'GENE_4', 'GENE_5']
corr = df[gene_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Gene correlations')
plt.show()
```

### 2.7.4 Step 4: Decide and Document

Based on Steps 1 to 3, decide:

- **What to drop?** (e.g. rows/columns with too many missing, non-informative)
- **What to impute?** (which columns, mean vs median vs mode)
- **What to scale?** (which features, and with what method, later)
- **What to fix?** (typos, units, encoding)

**Document your choices** in your notebook (comments or a short "Data cleaning checklist") so others, and future you, can reproduce your steps.

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

**Rule:** Run CV only on the training (and optionally validation) data. The test set is used once at the very end to report final performance.

**Important:** Split *before* scaling or other preprocessing that uses global statistics. Fit scalers (and similar) only on the training data, then apply them to validation and test. That keeps the test set truly unseen. We will practice this when we add scaling in later chapters.

## 2.9 Summary and Key Takeaways

**Python data structures:**  
**Lists** are ordered sequences (e.g. gene names, sample IDs). **Dictionaries** are key-value pairs (e.g. gene to expression). Use lists when order matters; use dictionaries when you look up by name.

**Pandas DataFrames:**  
You now use **DataFrames** for tables and **indexing** (`[]`, `.loc`, `.iloc`, boolean filters) to select columns and rows. These are the core tools for loading and manipulating biological data.

**Data quality:**  
Biological data often has missing values, outliers, batch effects, inconsistent units, and typos. We focused on **missing values** (drop vs impute with mean/median/mode) and **scaling** (why it matters, MinMax vs standardization).

**Basic statistics:**  
Use **mean**, **median**, **std**, **describe()**, and **correlation** to summarize and check your data before modeling.

**EDA workflow:**  
**Load → Summarize → Visualize → Decide.** Always document what you drop, impute, or transform.

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
- **`gene_expression.csv`**: Dummy dataset for this chapter, in the repo's `ML_book_easy/data/` folder. Download from GitHub and use it for all EDA and train/test examples above.  
- [UCI ML Repository](https://archive.ics.uci.edu/ml/): e.g. Breast Cancer Wisconsin dataset.  
- [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/): Gene expression data.

---

**End of Chapter 2**

In the next chapter, we introduce machine learning proper: supervised vs unsupervised learning, classification and regression, and key metrics such as accuracy, AUROC, and RMSE. We will train our first models using the clean data and train/validation/test split (and cross-validation when needed) you have just learned to prepare.
