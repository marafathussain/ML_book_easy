# Chapter 3: Data Cleaning and Exploratory Data Analysis (EDA)

## Introduction

In Chapter 2, you learned how to store data with **lists** and **dictionaries**. Now we put those building blocks to work. Before you train any machine learning model, you need to **clean** and **explore** your data. Clean, well-understood data is the foundation of useful ML. Messy or biased data leads to misleading or useless models. In biology, that often means missing samples, typos, batch effects, or wrong units, all of which can lead to wrong conclusions. So: spend time on data quality and exploration *before* you train models.

This chapter teaches you a simple, practical workflow: load your data, spot quality issues, handle missing values, normalize and scale when needed, summarize with basic statistics, and visualize. We also introduce the **train/test split**, which you will use from here on whenever you evaluate a model. By the end, you will know how to get biological data ready for machine learning, without going deep into theory.

We use **Pandas** for tables and **Matplotlib** or **Seaborn** for simple plots. All code runs in Google Colab; no installations required.

## 3.1 Pandas DataFrames and Indexing

Biological data usually comes as tables: rows are samples or genes, columns are features or measurements. In Python, we work with these tables using **Pandas DataFrames**. Think of a DataFrame as a spreadsheet: you can select columns, filter rows, and compute summaries.

### 3.1.1 What Is a DataFrame?

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

### 3.1.2 Selecting Columns and Rows (Indexing)

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

## 3.2 Data Quality Issues in Biological Data

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

## 3.3 Handling Missing Values

Missing values show up as `NaN` (Not a Number) in Pandas, or sometimes as blanks, "NA", or -999. First, see *where* and *how much* is missing:

```python
print(df.isna().sum())   # Count of missing per column
# Or
print(df.isnull().sum())
```

You have two main options: **remove** or **impute**.

### 3.3.1 Option 1: Remove (Drop)

- **Drop rows** with any missing: `df_clean = df.dropna()`
- **Drop columns** with too many missing (e.g. more than 20%):
  ```python
  threshold = len(df) * 0.2
  df_clean = df.dropna(thresh=len(df) - threshold, axis=1)
  ```

Use deletion when missingness is limited and you can afford to lose those rows or columns.

### 3.3.2 Option 2: Impute (Fill In)

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

## 3.4 Normalization and Scaling

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

## 3.5 Basic Statistics for Biological Data

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

## 3.6 Exploratory Data Analysis (EDA) Workflow

**EDA** means understanding your data *before* you build models. A simple workflow: **load → summarize → visualize → decide**.

### 3.6.1 Step 1: Load and Inspect

- Load the file: `pd.read_csv()`, `pd.read_excel()`, etc. (e.g. `gene_expression.csv` from the repo’s `ML_book_easy/data/` folder).
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

### 3.6.2 Step 2: Summarize

- **Missing:** `df.isna().sum()` or `df.isnull().sum()`
- **Numeric summary:** `df.describe()`
- **Categorical:** `df['Condition'].value_counts()`, `df['Condition'].nunique()`

Look for strange min/max, too many missing in one column, or unexpected categories.

### 3.6.3 Step 3: Visualize

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

### 3.6.4 Step 4: Decide and Document

Based on Steps 1 to 3, decide:

- **What to drop?** (e.g. rows/columns with too many missing, non-informative)
- **What to impute?** (which columns, mean vs median vs mode)
- **What to scale?** (which features, and with what method, later)
- **What to fix?** (typos, units, encoding)

**Document your choices** in your notebook (comments or a short “Data cleaning checklist”) so others, and future you, can reproduce your steps.

## 3.7 Train/Test Split: Introduction

To know how well a model **generalizes** to new data, we need to evaluate it on data it has *never* seen during training. That is why we split the dataset into a **training set** and a **test set**.

- **Training set:** Used to fit the model.
- **Test set:** Held out and used *only* to evaluate performance.

If you train and evaluate on the same data, you risk **overfitting** and overly optimistic results. The test set mimics “unseen” data.

**Rule:** Never train on the test set, and do not use it to choose models or tune parameters. Use it only for a final performance check.

### 3.7.1 How to Split

A common choice is **80% train, 20% test** (or 70/30, 90/10). Use **`train_test_split`** from scikit-learn and always set **`random_state`** so the split is reproducible:

```python
from sklearn.model_selection import train_test_split

# X = features, y = target (e.g. disease vs healthy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

This gives you four arrays: `X_train`, `X_test`, `y_train`, `y_test`. You fit the model on `X_train` and `y_train`, and evaluate on `X_test` and `y_test`. We use this pattern from the next chapter onward when we build our first ML models.

**Important:** Split *before* scaling or other preprocessing that uses global statistics. Fit scalers (and similar) only on the training data, then apply them to the test data. That keeps the test set truly “unseen.” We will practice this when we add scaling in later chapters.

## 3.8 Summary and Key Takeaways

**Python and Pandas:**  
You now use **DataFrames** for tables and **indexing** (`[]`, `.loc`, `.iloc`, boolean filters) to select columns and rows. These are the core tools for loading and manipulating biological data.

**Data quality:**  
Biological data often has missing values, outliers, batch effects, inconsistent units, and typos. We focused on **missing values** (drop vs impute with mean/median/mode) and **scaling** (why it matters, MinMax vs standardization).

**Basic statistics:**  
Use **mean**, **median**, **std**, **describe()**, and **correlation** to summarize and check your data before modeling.

**EDA workflow:**  
**Load → Summarize → Visualize → Decide.** Always document what you drop, impute, or transform.

**Train/test split:**  
Split data into **train** and **test** sets, use **`train_test_split`** with **`random_state`**, and evaluate only on the test set. Split *before* fitting scalers or other preprocessing.

With clean, explored data and a proper train/test split, you are ready to build and evaluate your first ML models in the next chapter.

## 3.9 Further Reading

**Pandas:**  
- [Pandas documentation](https://pandas.pydata.org/docs/): DataFrames, indexing, and data manipulation.

**Preprocessing:**  
- [Scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html): Scaling, encoding, and imputation.

**Visualization:**  
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/): Histograms, boxplots, heatmaps.

**Practice datasets:**  
- **`gene_expression.csv`**: Dummy dataset for this chapter, in the repo’s `ML_book_easy/data/` folder. Download from GitHub and use it for all EDA and train/test examples above.  
- [UCI ML Repository](https://archive.ics.uci.edu/ml/): e.g. Breast Cancer Wisconsin dataset.  
- [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/): Gene expression data.

---

**End of Chapter 3**

In the next chapter, we introduce machine learning proper: supervised vs unsupervised learning, classification and regression, and key metrics such as accuracy, AUROC, and RMSE. We will train our first models using the clean data and train/test split you have just learned to prepare.
