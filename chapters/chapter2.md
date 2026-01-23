# Chapter 2: Data Cleaning and Exploratory Data Analysis for Biological Data

## Introduction

In the world of machine learning, there's a popular saying: "garbage in, garbage out." This principle is especially true in biological research, where data quality can make or break your analysis. While the excitement of machine learning often centers on sophisticated algorithms and impressive predictions, experienced practitioners know that the real work—and often the most critical work—happens long before any model is trained.

This chapter focuses on the essential but often overlooked foundation of any successful machine learning project: data preparation and exploratory data analysis (EDA). We'll explore how to work with biological datasets, understand Python's core data structures, identify and fix data quality issues, and prepare your data for machine learning algorithms. By the end of this chapter, you'll understand why data scientists often spend 80% of their time on data preparation and only 20% on actual modeling.

## 2.1 Python Data Structures: Building Blocks for Biological Data Analysis

Before we dive into data cleaning and analysis, we need to understand how Python stores and organizes data. Think of data structures as containers—each designed for specific purposes, much like how a biologist uses different containers for different specimens.

### 2.1.1 Lists: Your First Data Container

Lists are Python's most basic sequential data structure. Imagine you're cataloging genes of interest from a recent study. A list lets you store these gene names in order:

**What is a list?**
A list is an ordered collection of items enclosed in square brackets `[]`. Items can be of any type—numbers, text (strings), or even other lists. Most importantly for our purposes, lists are ordered (the first item stays first) and mutable (you can change them after creation).

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
Suppose you're analyzing differential gene expression. You might have one list for upregulated genes and another for downregulated genes:

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
A dictionary stores data as key-value pairs enclosed in curly braces `{}`. Think of it like a real dictionary where you look up a word (key) to find its definition (value).

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

**Accessing values:**
You retrieve values using their keys:
- `brca1_info['gene_name']` returns 'BRCA1'
- `brca1_info['expression']` returns 5.2
- `brca1_info['pathways']` returns the list of pathways

**Adding and modifying:**
Dictionaries are mutable. You can add new information:
```python
brca1_info['fold_change'] = 2.5
```

Or modify existing values:
```python
brca1_info['expression'] = 5.8
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

Now you can access any gene's information:
```python
tp53_chromosome = gene_database['TP53']['chromosome']  # Returns 17
```

**When to use dictionaries:**
- Storing gene annotations where you look up by gene name
- Patient metadata where patient ID is the key
- Any scenario where meaningful labels are better than numeric positions
- When you need fast lookup by key (dictionaries are optimized for this)

### 2.1.3 Pandas DataFrames: The Biologist's Workhorse

While lists and dictionaries are fundamental, real biological data analysis requires something more powerful: the Pandas DataFrame. If lists are like single file folders and dictionaries are like filing cabinets, DataFrames are like entire spreadsheet applications.

**What is a DataFrame?**
A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. Think of it as an Excel spreadsheet or a SQL table in Python. It's the primary data structure you'll use for biological data analysis.

**Creating a simple DataFrame:**
```python
import pandas as pd

df = pd.DataFrame({
    'Gene': ['BRCA1', 'TP53', 'EGFR', 'MYC'],
    'Expression': [5.2, 12.8, 3.4, 7.1],
    'Chromosome': [17, 17, 7, 8],
    'Significant': [True, True, False, True]
})
```

This creates a table with four columns (Gene, Expression, Chromosome, Significant) and four rows (one per gene).

**DataFrame anatomy:**
- **Rows**: Represent individual observations (genes, samples, patients)
- **Columns**: Represent features or variables
- **Index**: Row labels (by default 0, 1, 2, 3... but can be customized)
- **Column names**: Descriptive headers for each variable

**Why DataFrames are essential for biology:**

1. **Mixed data types**: A single DataFrame can contain gene names (text), expression values (numbers), and significance flags (boolean), just like real biological data.

2. **Labeled axes**: Both rows and columns have meaningful names, making code readable and reducing errors.

3. **Built-in functions**: Pandas provides hundreds of functions for data manipulation, statistical analysis, and visualization.

4. **Integration**: Works seamlessly with machine learning libraries like scikit-learn and visualization libraries like matplotlib.

**Real-world example - RNA-seq dataset:**

Imagine you've performed RNA-seq on 100 patients—60 healthy and 40 with disease. For each patient, you've measured expression of 50 genes. Your DataFrame might look like:

```
   Sample_ID  Condition  Age  Gender  GENE_1  GENE_2  GENE_3  ...  GENE_50
0  SAMPLE_001  Healthy   45    M      5.2     3.8     12.1   ...   4.5
1  SAMPLE_002  Disease   62    F      8.9     4.2     15.3   ...   6.8
2  SAMPLE_003  Healthy   38    M      4.8     3.5     11.8   ...   4.2
...
99 SAMPLE_100  Disease   71    F      9.2     4.5     16.1   ...   7.1
```

Each row represents one patient, and each column represents either metadata (Sample_ID, Condition, Age, Gender) or gene expression measurements.

### 2.1.4 Pandas Indexing: Selecting Your Data

The power of DataFrames lies in flexible data selection. Pandas offers multiple ways to slice and dice your data.

**Column selection:**

To select a single column, use square brackets with the column name:
```python
df['Gene']  # Returns a Series (one-dimensional)
```

To select multiple columns, use a list of column names:
```python
df[['Gene', 'Expression']]  # Returns a DataFrame (two-dimensional)
```

**Row selection by position (.iloc):**

`.iloc` selects rows and columns by integer position (like list indexing):

```python
df.iloc[0]           # First row (as Series)
df.iloc[0:3]         # First three rows
df.iloc[0, 1]        # Row 0, Column 1 (single value)
df.iloc[:, 0:2]      # All rows, first two columns
```

**Row selection by label (.loc):**

`.loc` selects rows and columns by labels:

```python
df.loc[0]                      # Row with index 0
df.loc[0:2, 'Gene']           # Rows 0-2, Gene column
df.loc[:, 'Gene':'Expression'] # All rows, columns from Gene to Expression
```

**Boolean indexing (the most powerful!):**

This is where Pandas truly shines for biological data analysis. You can filter rows based on conditions:

```python
# All genes with expression > 5.0
high_expression = df[df['Expression'] > 5.0]

# Only significant genes
significant_genes = df[df['Significant'] == True]

# Multiple conditions: disease samples AND age > 50
older_disease = df[(df['Condition'] == 'Disease') & (df['Age'] > 50)]

# Genes on chromosome 17
chr17_genes = df[df['Chromosome'] == 17]
```

**Real biological example:**

Suppose you want to find genes that are:
1. Significantly differentially expressed
2. Have expression greater than 10
3. Are located on chromosome 17

```python
candidate_genes = df[
    (df['Significant'] == True) & 
    (df['Expression'] > 10) & 
    (df['Chromosome'] == 17)
]
```

This type of filtering is fundamental to biological data analysis—identifying genes or samples that meet specific criteria.

## 2.2 Data Quality Issues in Biological Datasets

Real-world biological data is messy. Unlike the clean datasets in textbooks, actual experimental data comes with numerous quality issues. Understanding these problems is the first step in addressing them.

### 2.2.1 Missing Values: The Unavoidable Reality

Missing values are perhaps the most common data quality issue in biological research. They occur for many reasons, and how you handle them can significantly impact your analysis.

**Why do missing values occur?**

1. **Technical failures**: A sample didn't amplify in PCR, a microarray spot was defective, or a sequencing read failed quality control.

2. **Below detection limit**: Gene expression too low to be reliably measured. In RNA-seq, genes with very low read counts are often filtered out.

3. **Random chance**: Sometimes instruments fail, reagents degrade, or human error occurs.

4. **Study design**: Not all measurements were taken for all samples (e.g., some clinical variables only measured in disease patients).

**How missing values appear in data:**

Missing values can be represented in many ways:
- Python's `None` or `NaN` (Not a Number)
- Empty cells in spreadsheets
- Special codes like -999, 0, or "NA"
- Literally the text "missing" or "N/A"

This inconsistency is why data cleaning is crucial—you need to standardize how missing values are represented before analysis.

**Types of missingness:**

Understanding why data is missing helps you decide how to handle it:

1. **Missing Completely at Random (MCAR)**: The probability of being missing is the same for all observations. Example: A batch of samples was lost due to freezer malfunction—unrelated to any biological or technical factors.

2. **Missing at Random (MAR)**: The probability of being missing is related to other observed variables but not the missing value itself. Example: Older patients were less likely to have certain tests performed, but among patients of the same age, missingness is random.

3. **Missing Not at Random (MNAR)**: The probability of being missing is related to the unobserved value itself. Example: Gene expression below detection limit is missing precisely because it's low. This is the most problematic type.

**Example in RNA-seq data:**

RNA-seq commonly has 5-15% missing values. Genes with very low expression might have:
- Sample 1: 0.2 (measured)
- Sample 2: NaN (below detection)
- Sample 3: 0.1 (measured)
- Sample 4: NaN (below detection)

The missingness is MNAR—it's missing because expression is low, which is exactly what we're trying to measure!

### 2.2.2 Outliers: Signal or Noise?

Outliers are extreme values that differ substantially from other observations. In biology, the challenge is distinguishing between:
- **Technical errors**: Measurement mistakes, contamination, equipment malfunction
- **Biological truth**: Genuine extreme biological variation

**Example scenario:**

You're measuring gene expression across 100 samples:
- 99 samples: expression between 2.0 and 8.0
- 1 sample: expression of 150.0

Is this:
- A measurement error (sample was contaminated)?
- A data entry error (should be 15.0)?
- Real biology (this patient has a gene amplification)?

**How to identify outliers:**

1. **Visualization**: Box plots show outliers as points beyond the "whiskers"

2. **Statistical methods**:
   - Values beyond mean ± 3 standard deviations
   - Values beyond 1.5 × IQR (interquartile range) from quartiles
   - Z-scores with |z| > 3

3. **Domain knowledge**: Does this value make biological sense?

**Handling outliers:**

- **Investigate**: Look at the raw data, lab notes, sample quality metrics
- **Transform**: Log transformation can reduce the impact of extreme values
- **Cap**: Replace extreme values with a threshold (winsorization)
- **Remove**: Only if you're confident it's an error and have documentation

**Important principle**: Never remove outliers simply to improve model performance without biological justification. That outlier might be your most interesting discovery!

### 2.2.3 Inconsistent Formatting

Biological data often comes from multiple sources, databases, or experiments, leading to inconsistent formatting.

**Common examples:**

1. **Gene names**:
   - "BRCA1" vs. "brca1" vs. "Brca1" vs. "BRCA-1"
   - Hugo symbols vs. Ensembl IDs vs. Entrez IDs

2. **Species names**:
   - "Homo sapiens" vs. "Human" vs. "H. sapiens" vs. "human"

3. **Dates**:
   - "2024-01-15" vs. "01/15/2024" vs. "15-Jan-2024"

4. **Categorical variables**:
   - Disease status: "disease" vs. "Disease" vs. "diseased" vs. "1"
   - Gender: "M/F" vs. "Male/Female" vs. "1/2"

**Why it matters:**

Python treats "BRCA1" and "brca1" as completely different strings. If you're filtering for "BRCA1", you'll miss any entries recorded as "brca1".

**Solutions:**

1. **Stratified sampling**: Ensure both classes in train/test
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y, test_size=0.2, random_state=42
   )
   ```

2. **Resampling**:
   - **Oversampling minority class**: Duplicate disease samples
   - **Undersampling majority class**: Remove some healthy samples
   - **SMOTE**: Synthetic Minority Over-sampling Technique (creates synthetic disease samples)

   ```python
   from imblearn.over_sampling import SMOTE
   
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Class weights**: Penalize misclassifying minority class more heavily
   ```python
   from sklearn.linear_model import LogisticRegression
   
   model = LogisticRegression(class_weight='balanced')
   ```

4. **Use appropriate metrics**: Don't rely on accuracy alone
   - Precision: Of predicted disease, how many truly have disease?
   - Recall: Of actual disease cases, how many did we catch?
   - F1-score: Harmonic mean of precision and recall
   - ROC-AUC: Area under ROC curve

**Example:**

Dataset: 90 healthy, 10 disease

**Model A** (predicts "healthy" always):
- Accuracy: 90% (misleading!)
- Recall for disease: 0% (caught none!)
- F1-score: 0 (reveals true performance)

**Model B** (proper classifier):
- Accuracy: 85%
- Recall for disease: 70% (caught 7/10)
- F1-score: 0.58 (more honest assessment)

Model B is clearly better despite lower accuracy!

### 2.8.4 Time Series and Longitudinal Data

Some biological data has temporal structure:
- Patient measurements over time
- Gene expression at different time points
- Sequential drug responses

**Special considerations:**

1. **Don't randomize time points**: Keep temporal order intact
2. **Time-based splitting**: 
   - Train on early time points
   - Test on later time points
   
   ```python
   # Instead of random split
   split_point = int(0.8 * len(df))
   train = df[:split_point]
   test = df[split_point:]
   ```

3. **Account for autocorrelation**: Measurements close in time are more similar

4. **Patient-level splitting**: If multiple time points per patient, keep all time points for a patient together
   ```python
   # Group by patient
   patients = df['Patient_ID'].unique()
   train_patients, test_patients = train_test_split(patients, test_size=0.2)
   
   train = df[df['Patient_ID'].isin(train_patients)]
   test = df[df['Patient_ID'].isin(test_patients)]
   ```

### 2.8.5 Multi-omics Data Integration

Modern biology often combines multiple data types:
- Genomics (DNA variants)
- Transcriptomics (gene expression)
- Proteomics (protein levels)
- Metabolomics (metabolite concentrations)

**Challenges:**

1. **Different scales**: Gene expression might range 0-1000, metabolites 0-10
2. **Different dimensions**: 20,000 genes, 500 metabolites
3. **Different distributions**: Some log-normal, others normal
4. **Missing patterns differ**: Each omics type has different missingness

**Approaches:**

1. **Scale each omics type separately**:
   ```python
   # Standardize each omics type
   genomics_scaled = StandardScaler().fit_transform(genomics_data)
   proteomics_scaled = StandardScaler().fit_transform(proteomics_data)
   
   # Concatenate
   X_combined = np.hstack([genomics_scaled, proteomics_scaled])
   ```

2. **Late fusion**: Train separate models for each omics, combine predictions

3. **Feature selection per omics**: Select top features from each before combining

4. **Specialized algorithms**: Multi-view learning, network-based integration

## 2.9 Common Mistakes and How to Avoid Them

### 2.9.1 Mistake: Looking at Test Set During Development

**What people do:**
```python
# Train model
model.fit(X_train, y_train)

# Check test performance
test_score = model.score(X_test, y_test)  # 75%

# "That's not good enough, let me try different features"
# ... selects features based on test performance ...

# Try again
model2.fit(X_train_new, y_train)
test_score2 = model2.score(X_test_new, y_test)  # 85%
```

**Why it's wrong:**

You've now used the test set to make modeling decisions! It's no longer truly unseen. Your reported 85% is optimistic.

**Correct approach:**

Use cross-validation on training set for all development decisions. Touch test set only once at the very end.

```python
# Use cross-validation on training set
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean()}")  # Use this to guide development

# Only at the very end, evaluate on test set once
final_score = model.score(X_test, y_test)
```

### 2.9.2 Mistake: Not Documenting Data Cleaning Steps

**The problem:**

You clean data interactively in Jupyter notebook:
- Drop some outliers
- Impute some missing values
- Remove some samples
- Transform some features

Six months later, you can't remember exactly what you did. Reviewers ask questions you can't answer.

**Solution:**

Document every step in code with comments and save intermediate files:

```python
# Load original data
df_original = pd.read_csv('raw_data.csv')
df_original.to_csv('data_00_original.csv', index=False)

# Remove samples with >50% missing values
df_step1 = df_original.dropna(thresh=len(df_original.columns)*0.5)
df_step1.to_csv('data_01_removed_missing_samples.csv', index=False)
print(f"Removed {len(df_original) - len(df_step1)} samples")

# Remove outliers (expression > 3 SD from mean)
mean = df_step1['GENE_1'].mean()
std = df_step1['GENE_1'].std()
df_step2 = df_step1[df_step1['GENE_1'] < mean + 3*std]
df_step2.to_csv('data_02_removed_outliers.csv', index=False)
print(f"Removed {len(df_step1) - len(df_step2)} outliers")

# Final clean data
df_clean = df_step2
df_clean.to_csv('data_03_final_clean.csv', index=False)
```

Better yet, create a cleaning function:

```python
def clean_gene_expression_data(df, missing_threshold=0.5, outlier_sd=3):
    """
    Clean gene expression data.
    
    Parameters:
    -----------
    df : DataFrame
        Raw gene expression data
    missing_threshold : float
        Remove samples with more than this proportion missing
    outlier_sd : float
        Remove values beyond this many standard deviations
        
    Returns:
    --------
    DataFrame : Cleaned data
    """
    # Your cleaning steps here
    # Document each step clearly
    
    return df_clean
```

### 2.9.3 Mistake: Treating Technical Replicates as Independent Samples

**The scenario:**

You measure the same biological sample three times (technical replicates):
- Sample_A_rep1
- Sample_A_rep2  
- Sample_A_rep3

If you treat these as three independent samples:
- Your sample size is inflated (n=3 instead of n=1)
- Violations of independence assumptions
- Overly optimistic standard errors

**Correct approaches:**

1. **Average the replicates**:
   ```python
   df_averaged = df.groupby('Sample_ID').mean()
   ```

2. **Use one replicate** (if quality is similar):
   ```python
   df_single = df.groupby('Sample_ID').first()
   ```

3. **Model the correlation structure** (advanced):
   Use mixed-effects models that account for repeated measures

**How to identify:**

Check your sample IDs. Do you see patterns like:
- Patient_001_A, Patient_001_B, Patient_001_C
- Sample_01_rep1, Sample_01_rep2
- TCGA_001.tumor, TCGA_001.normal

These indicate related samples that need special handling!

### 2.9.4 Mistake: Using Mean for Skewed Data

**The problem:**

Gene expression values: [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]
- Mean: 12.7 (pulled up by outlier)
- Median: 3 (representative of typical value)

If you impute missing values with mean=12.7, you're filling in values much higher than most of the data!

**Better approach:**

```python
# Check skewness
from scipy import stats
skewness = stats.skew(df['GENE_1'])

if abs(skewness) > 1:  # Highly skewed
    # Use median
    df['GENE_1'].fillna(df['GENE_1'].median(), inplace=True)
else:
    # Use mean
    df['GENE_1'].fillna(df['GENE_1'].mean(), inplace=True)
```

Or even better for gene expression:

```python
# Log transform first, then impute
df['GENE_1_log'] = np.log2(df['GENE_1'] + 1)
df['GENE_1_log'].fillna(df['GENE_1_log'].mean(), inplace=True)
# Transform back if needed
df['GENE_1_imputed'] = 2**df['GENE_1_log'] - 1
```

### 2.9.5 Mistake: Ignoring Data Types

**Common issue:**

```python
df = pd.read_csv('data.csv')
print(df['Age'].mean())  # Error: can't calculate mean of object type!
```

The 'Age' column was read as text because of some non-numeric values ("N/A", "Unknown").

**Solution:**

```python
# Inspect data types
print(df.dtypes)

# Check for non-numeric values
print(df[pd.to_numeric(df['Age'], errors='coerce').isna()])

# Convert, coercing errors to NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Then handle NaN values appropriately
df['Age'].fillna(df['Age'].median(), inplace=True)
```

**Categorical variables:**

Don't forget to properly encode categorical variables before modeling:

```python
# Wrong: treating "Tissue_Type" as if it were numeric
X = df[['Tissue_Type', 'Gene_1', 'Gene_2']]  # Tissue_Type is text!

# Correct: one-hot encode
X = pd.get_dummies(df[['Tissue_Type', 'Gene_1', 'Gene_2']], 
                   columns=['Tissue_Type'])
```

## 2.10 Putting It All Together: A Complete Example

Let's walk through a complete, realistic example from start to finish.

### 2.10.1 The Scenario

You've received RNA-seq data from a collaborator studying breast cancer. The dataset contains:
- 120 patient samples (80 cancer, 40 healthy)
- 500 genes selected as potentially relevant
- Patient metadata: age, tumor stage, treatment status

Your goal: Build a classifier to distinguish cancer from healthy tissue based on gene expression.

### 2.10.2 Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('breast_cancer_expression.csv')

# Initial inspection
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes.value_counts())

print("\nBasic info:")
print(df.info())

# Check target distribution
print("\nClass distribution:")
print(df['Diagnosis'].value_counts())
print(f"Imbalance ratio: {80/40:.1f}:1")
```

**Output:**
```
Dataset shape: (120, 505)  # 120 samples, 505 features

Class distribution:
Cancer     80
Healthy    40
Imbalance ratio: 2.0:1
```

### 2.10.3 Data Quality Assessment

```python
# Check for missing values
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_summary = pd.DataFrame({
    'Count': missing,
    'Percentage': missing_pct
})
print(missing_summary[missing_summary['Count'] > 0].sort_values('Count', ascending=False))

# Visualize missing data
plt.figure(figsize=(10, 6))
missing_pct[missing_pct > 0].plot(kind='bar')
plt.title('Missing Data Percentage by Feature')
plt.ylabel('Percentage Missing')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('missing_data.png')
plt.show()

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Check for duplicate patient IDs
duplicate_ids = df['Patient_ID'].duplicated().sum()
print(f"Duplicate Patient IDs: {duplicate_ids}")
```

**Findings:**
- 15 genes have 5-10% missing values
- 2 genes have 25% missing values
- No duplicate rows
- No duplicate patient IDs

### 2.10.4 Data Cleaning

```python
# Remove genes with >20% missing values
gene_columns = [col for col in df.columns if col.startswith('GENE_')]
missing_per_gene = df[gene_columns].isnull().sum() / len(df)
genes_to_drop = missing_per_gene[missing_per_gene > 0.2].index.tolist()

print(f"Dropping {len(genes_to_drop)} genes with >20% missing: {genes_to_drop}")
df = df.drop(columns=genes_to_drop)

# Update gene list
gene_columns = [col for col in df.columns if col.startswith('GENE_')]
print(f"Remaining genes: {len(gene_columns)}")

# Impute remaining missing values with median
for gene in gene_columns:
    if df[gene].isnull().sum() > 0:
        median_val = df[gene].median()
        df[gene].fillna(median_val, inplace=True)
        print(f"Imputed {gene} with median {median_val:.2f}")

# Verify no missing values remain
print(f"\nTotal missing values after cleaning: {df.isnull().sum().sum()}")

# Handle missing metadata
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Stage'] = df['Stage'].fillna('Unknown')

# Save cleaned data
df.to_csv('data_cleaned.csv', index=False)
```

### 2.10.5 Exploratory Data Analysis

```python
# Summary statistics by diagnosis
print("\nAge by Diagnosis:")
print(df.groupby('Diagnosis')['Age'].describe())

# Visualize age distribution
plt.figure(figsize=(10, 5))
df.boxplot(column='Age', by='Diagnosis')
plt.ylabel('Age')
plt.title('Age Distribution by Diagnosis')
plt.suptitle('')  # Remove default title
plt.savefig('age_distribution.png')
plt.show()

# Gene expression distributions
# Select a few genes to visualize
sample_genes = gene_columns[:5]

fig, axes = plt.subplots(1, len(sample_genes), figsize=(15, 4))
for i, gene in enumerate(sample_genes):
    axes[i].hist(df[gene], bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_xlabel('Expression')
    axes[i].set_title(gene)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('gene_distributions.png')
plt.show()

# Check for skewness - decide if log transform needed
skewness_values = df[sample_genes].skew()
print("\nSkewness of sample genes:")
print(skewness_values)
print("\nNote: |skew| > 1 suggests log transformation may help")

# Correlation between first 10 genes
correlation_matrix = df[gene_columns[:10]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True)
plt.title('Gene Expression Correlations (First 10 Genes)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Compare cancer vs healthy for top genes
top_genes = gene_columns[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, gene in enumerate(top_genes):
    df.boxplot(column=gene, by='Diagnosis', ax=axes[i])
    axes[i].set_ylabel('Expression Level')
    axes[i].set_title(f'{gene} by Diagnosis')
    axes[i].get_figure().suptitle('')
plt.tight_layout()
plt.savefig('genes_by_diagnosis.png')
plt.show()
```

**Observations:**
- Age is similar between groups (good - not a confounding factor)
- Gene expression is right-skewed (log transformation may help)
- Some genes show clear separation between cancer and healthy
- High correlation between some genes (might be redundant)

### 2.10.6 Feature Engineering

```python
# Log transformation for skewed genes
# Identify highly skewed genes (|skew| > 1)
skewed_genes = []
for gene in gene_columns:
    skew = df[gene].skew()
    if abs(skew) > 1:
        skewed_genes.append(gene)
        df[f'{gene}_log'] = np.log2(df[gene] + 1)  # +1 to handle zeros

print(f"Applied log transformation to {len(skewed_genes)} genes")

# Update gene list to include log-transformed versions
log_gene_columns = [f'{gene}_log' for gene in skewed_genes]
all_gene_features = log_gene_columns

# Encode categorical variables
df['Stage_encoded'] = df['Stage'].map({
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'Unknown': 0
})

# Create age groups (optional, for interpretability)
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[0, 40, 60, 100],
                         labels=['Young', 'Middle', 'Senior'])
```

### 2.10.7 Prepare for Modeling

```python
# Define features and target
feature_columns = all_gene_features + ['Age', 'Stage_encoded']
X = df[feature_columns]
y = df['Diagnosis']

# Encode target (Cancer=1, Healthy=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Cancer=0, Healthy=1 or vice versa
print(f"Encoding: {le.classes_}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,          # 20% test (24 samples)
    random_state=42,        # Reproducibility
    stratify=y_encoded      # Maintain 2:1 ratio
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"\nTrain class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nTest class distribution:")
print(pd.Series(y_test).value_counts())

# Scale features (FIT on train, TRANSFORM on both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics!

# Convert back to DataFrame for readability (optional)
X_train_scaled = pd.DataFrame(
    X_train_scaled, 
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns, 
    index=X_test.index
)

# Verify scaling worked correctly
print("\nScaling verification (GENE_1_log):")
print(f"Training: mean={X_train_scaled['GENE_1_log'].mean():.4f}, std={X_train_scaled['GENE_1_log'].std():.4f}")
print(f"Test: mean={X_test_scaled['GENE_1_log'].mean():.4f}, std={X_test_scaled['GENE_1_log'].std():.4f}")
print("Training should be ~0 mean, ~1 std. Test will differ slightly.")

# Save processed data for modeling
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nData preparation complete!")
print("Ready for machine learning modeling (Week 3)!")
```

**Final Summary:**
```
Original data: 120 samples, 500 genes
After cleaning: 120 samples, 498 genes
After feature engineering: 120 samples, 350 features
Train set: 96 samples (64 cancer, 32 healthy)
Test set: 24 samples (16 cancer, 8 healthy)
Class balance maintained: 2:1 ratio in both sets
All features scaled: mean≈0, std≈1
No data leakage: Scaling fitted only on training data
```

## 2.11 Summary and Key Takeaways

This chapter covered the essential foundations of data preparation for machine learning in biology:

**Core Concepts:**

1. **Python data structures**: Lists for sequences, dictionaries for key-value pairs, DataFrames for tabular data

2. **Data quality issues**: Missing values, outliers, inconsistent formatting, duplicates, and batch effects are the norm in biological data

3. **Missing value strategies**: Choose between deletion, imputation, or modeling based on amount and type of missingness

4. **Normalization and scaling**: Essential for most ML algorithms; choose Min-Max for bounded output, Standardization for most cases

5. **Basic statistics**: Use mean, median, standard deviation, and correlation to understand your data before modeling

6. **EDA workflow**: Load → Clean → Explore → Visualize → Engineer → Prepare

7. **Train/test split**: The foundation of honest evaluation; always split before scaling to avoid data leakage

**Biological Data Challenges:**

- High dimensionality (p >> n)
- Batch effects
- Class imbalance
- Non-normal distributions
- Multi-omics integration

**Critical Rules to Remember:**

1. **Never look at test data during development** - it's sacred
2. **Always split before scaling** - fit transformations on training only  
3. **Document every cleaning step** - reproducibility is essential
4. **Stratify for imbalanced data** - maintain class proportions
5. **Choose imputation based on missingness type** - MCAR, MAR, or MNAR
6. **Visualize before and after** - ensure transformations worked as expected

**Common Mistakes:**

- Scaling before splitting (data leakage!)
- Treating technical replicates as independent samples
- Using mean for highly skewed data
- Removing outliers to improve performance without biological justification
- Not documenting data cleaning decisions

**Next Steps:**

With clean, properly prepared data, you're now ready to:
- Apply machine learning algorithms (Week 3)
- Evaluate model performance with appropriate metrics
- Interpret results in biological context
- Deploy models for real-world predictions

Remember: In machine learning, "garbage in, garbage out" is not just a saying—it's a fundamental truth. The time you invest in proper data preparation will pay dividends in model performance and scientific validity.

## 2.12 Further Reading and Resources

**Books:**
- McKinney, W. (2022). *Python for Data Analysis, 3rd Edition*. O'Reilly.
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly.

**Papers:**
- Leek, J. T., et al. (2010). Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics*, 11(10), 733-739.
- Little, R. J., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data, 3rd Edition*. Wiley.

**Online Resources:**
- Pandas documentation: https://pandas.pydata.org/docs/
- Scikit-learn preprocessing guide: https://scikit-learn.org/stable/modules/preprocessing.html
- Seaborn tutorial: https://seaborn.pydata.org/tutorial.html

**Practice Datasets:**
- Gene Expression Omnibus (GEO): https://www.ncbi.nlm.nih.gov/geo/
- The Cancer Genome Atlas (TCGA): https://portal.gdc.cancer.gov/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/

---

**End of Chapter 2**

In the next chapter, we'll use this cleaned and prepared data to build our first machine learning models, starting with logistic regression for classification of disease states based on gene expression profiles.. **Standardize text case**:
```python
df['Gene'] = df['Gene'].str.upper()  # Convert all to uppercase
```

2. **Strip whitespace**:
```python
df['Gene'] = df['Gene'].str.strip()  # Remove leading/trailing spaces
```

3. **Replace variations**:
```python
df['Species'] = df['Species'].replace({
    'human': 'Homo sapiens',
    'Human': 'Homo sapiens',
    'H. sapiens': 'Homo sapiens'
})
```

4. **Use standardized identifiers**: Prefer stable database IDs over gene symbols when possible.

### 2.2.4 Duplicate Records

Duplicates occur when the same observation appears multiple times in your dataset.

**How duplicates arise:**

1. **Data merging**: Combining datasets from different sources
2. **Technical replicates**: Same sample measured multiple times (should be labeled as replicates, not independent samples)
3. **Data entry errors**: Accidentally importing the same file twice
4. **Biological replicates**: Different samples from the same individual

**Why duplicates are problematic:**

1. **Inflated sample size**: You think you have 100 independent samples, but actually have 80 unique samples measured multiple times.

2. **Biased statistics**: Duplicates artificially strengthen certain patterns.

3. **Data leakage**: If the same sample appears in both training and test sets, your model has "seen" the test data!

**Detecting duplicates:**
```python
# Check for duplicate rows
duplicates = df.duplicated().sum()

# Find which rows are duplicates
duplicate_rows = df[df.duplicated()]

# Check for duplicates based on specific columns
sample_duplicates = df.duplicated(subset=['Sample_ID']).sum()
```

**Handling duplicates:**

1. **Exact duplicates**: Usually safe to remove
```python
df_clean = df.drop_duplicates()
```

2. **Technical replicates**: Average the measurements
```python
df_averaged = df.groupby('Sample_ID').mean()
```

3. **Biological replicates**: Keep separate but properly labeled

### 2.2.5 Batch Effects

Batch effects are systematic differences between groups of samples processed at different times or by different methods. This is one of the most insidious problems in biological data.

**What causes batch effects?**

1. **Different sequencing runs**: RNA-seq performed on different dates
2. **Different laboratories**: Multi-center studies
3. **Different reagent lots**: Subtle differences in reagent composition
4. **Different technicians**: Variations in handling techniques
5. **Equipment calibration**: Instruments drift over time

**Example:**

Imagine a cancer study with samples from two hospitals:
- Hospital A samples: All cluster together in PCA plot
- Hospital B samples: All cluster together in PCA plot
- Healthy vs. disease status: No clear separation

The strongest signal in your data is hospital (batch), not disease! Your machine learning model will learn to predict hospital, not cancer.

**Detecting batch effects:**

1. **Visualization**: PCA plots colored by batch
2. **Statistical tests**: ANOVA testing for batch differences
3. **Correlation analysis**: Samples from same batch more similar than expected

**Correcting batch effects:**

1. **ComBat**: Empirical Bayes method for batch correction
2. **Surrogate Variable Analysis (SVA)**: Identifies and removes hidden batch effects
3. **Experimental design**: Balance batches (include healthy and disease in each batch)
4. **Include batch as covariate**: Let your model account for batch effects

**Important note**: Batch correction is a complex topic with many methods and considerations. The key is to detect and account for batch effects rather than ignore them.

## 2.3 Handling Missing Values: Strategies and Trade-offs

Now that we understand why missing values occur, let's explore strategies for handling them. Each approach has advantages and disadvantages.

### 2.3.1 Strategy 1: Deletion

The simplest approach is to remove missing values from your dataset.

**Listwise deletion (complete case analysis):**
Remove any row that has any missing value.

```python
df_clean = df.dropna()
```

**Advantages:**
- Simple and straightforward
- No assumptions about missing data
- Preserves observed relationships

**Disadvantages:**
- Can lose substantial data (if any column has missing values, entire row is removed)
- Reduces statistical power
- Can introduce bias if data is not MCAR

**When to use:**
- Missing values are rare (<5%)
- You have abundant data
- Missing data is MCAR
- Simplicity is more important than sample size

**Column-wise deletion:**
Remove columns (features) with too many missing values.

```python
# Remove columns with more than 20% missing
threshold = len(df) * 0.2
df_clean = df.dropna(thresh=len(df) - threshold, axis=1)
```

**When to use:**
- A few variables have extensive missingness
- Those variables are not critical to your analysis
- You have many features and can afford to lose some

**Practical example:**

You have 100 samples and 50 genes. Gene_42 is missing in 60 samples (60% missing). Rather than losing 60 samples (if you drop rows), you drop Gene_42 (1 gene) and keep all 100 samples for the remaining 49 genes.

### 2.3.2 Strategy 2: Imputation

Imputation means filling in missing values with estimated values.

**Mean/Median imputation:**

Replace missing values with the mean (for normally distributed data) or median (for skewed data or when outliers are present).

```python
# Mean imputation
df['GENE_1'].fillna(df['GENE_1'].mean(), inplace=True)

# Median imputation (more robust to outliers)
df['GENE_1'].fillna(df['GENE_1'].median(), inplace=True)
```

**Advantages:**
- Retains all samples
- Simple to implement
- Reasonable for small amounts of missing data

**Disadvantages:**
- Underestimates variance (all imputed values are the same)
- Distorts relationships between variables
- Doesn't account for correlations

**When to use:**
- Missing values are moderate (5-30%)
- Data is MAR
- You need to retain sample size
- As a quick first approach

**Mode imputation (for categorical variables):**

For categorical data, use the most common value.

```python
# Mode imputation for tissue type
most_common_tissue = df['Tissue'].mode()[0]
df['Tissue'].fillna(most_common_tissue, inplace=True)
```

**Forward/Backward fill (for time series):**

Use the previous (forward fill) or next (backward fill) observation.

```python
# Forward fill - use previous time point's value
df['Temperature'].fillna(method='ffill', inplace=True)

# Backward fill - use next time point's value
df['Temperature'].fillna(method='bfill', inplace=True)
```

**When to use:**
- Time series data where values change gradually
- Measurements taken at regular intervals
- Missing values are isolated (not consecutive)

**K-Nearest Neighbors (KNN) imputation:**

Use values from similar samples to impute missing data.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[gene_columns]),
    columns=gene_columns
)
```

**How it works:**
1. Find the 5 most similar samples (based on non-missing features)
2. Average their values for the missing feature
3. Use this average as the imputed value

**Advantages:**
- Accounts for relationships between variables
- More sophisticated than simple mean
- Often performs well in practice

**Disadvantages:**
- Computationally intensive for large datasets
- Requires choosing number of neighbors
- Still makes assumptions about missing data

**When to use:**
- Moderate missing data (10-40%)
- Features are correlated
- You have enough computational resources
- MAR assumption holds

**Multiple Imputation:**

Create multiple imputed datasets, analyze each, and combine results.

**How it works:**
1. Generate 5-10 different imputed datasets (each with slightly different imputed values)
2. Run your analysis on each dataset separately
3. Combine results using established statistical rules

**Advantages:**
- Properly accounts for imputation uncertainty
- Gold standard for missing data in statistics
- Provides valid statistical inference

**Disadvantages:**
- Complex to implement
- Computationally intensive
- Requires additional statistical knowledge

**When to use:**
- Statistical inference is critical (p-values, confidence intervals)
- Missing data is substantial
- You need publication-quality results
- Have statistical expertise

### 2.3.3 Strategy 3: Keep Missing Values

Some approaches don't require imputation.

**Missing indicator method:**

Create a new binary variable indicating whether data was missing.

```python
# Create missing indicator
df['GENE_1_missing'] = df['GENE_1'].isna().astype(int)

# Then impute original variable (e.g., with mean)
df['GENE_1'].fillna(df['GENE_1'].mean(), inplace=True)
```

**Advantage:**
- Preserves information about missingness
- Model can learn if missingness is informative

**When to use:**
- Missingness might be MNAR
- The fact that data is missing could be meaningful
- Example: Certain tests only performed on sicker patients

**Algorithms that handle missing values:**

Some machine learning algorithms can work with missing values directly:
- XGBoost
- LightGBM
- Random Forests (some implementations)

**When to use:**
- You're using these specific algorithms
- Missing data patterns are complex
- You want the algorithm to learn optimal treatment of missing values

### 2.3.4 Choosing a Strategy

**Decision framework:**

1. **How much is missing?**
   - <5%: Deletion is usually fine
   - 5-30%: Consider imputation
   - 30-60%: Advanced imputation or specialized methods
   - >60%: Consider dropping the variable

2. **Why is it missing?**
   - MCAR: Most methods work
   - MAR: Imputation methods that use other variables
   - MNAR: Missing indicator or specialized methods

3. **What's your goal?**
   - Prediction: Simpler imputation often sufficient
   - Inference: More sophisticated methods needed
   - Exploration: Deletion may be acceptable

4. **How much data do you have?**
   - Large dataset: Can afford deletion
   - Small dataset: Imputation to preserve sample size

5. **How important is the variable?**
   - Critical variable: Invest in good imputation
   - Less important: Simple imputation or deletion

**Practical recommendation for biologists:**

For most biological datasets with moderate missing data (5-20%):
1. Drop columns with >20-30% missing
2. Use median imputation for remaining numerical variables
3. Use mode imputation for categorical variables
4. Document your approach clearly

For high-stakes projects or publications:
1. Investigate patterns of missingness
2. Use multiple imputation or KNN imputation
3. Perform sensitivity analyses comparing different imputation methods
4. Report how results change under different missing data assumptions

## 2.4 Normalization and Scaling: Making Features Comparable

Different genes have vastly different expression ranges. Some genes are highly expressed (thousands of reads), others barely expressed (single-digit reads). Machine learning algorithms can be misled by these scale differences.

### 2.4.1 Why Normalization Matters

**The problem:**

Imagine two genes in your dataset:
- Gene A: Expression ranges from 0 to 10
- Gene B: Expression ranges from 0 to 10,000

When you feed this into a machine learning algorithm that calculates distances (like K-Nearest Neighbors or K-Means clustering), Gene B dominates the calculation simply because its numbers are larger—even if Gene A is more biologically relevant for your question!

**Mathematical illustration:**

Distance between two samples using Euclidean distance:
```
Sample 1: Gene_A = 5,  Gene_B = 5000
Sample 2: Gene_A = 6,  Gene_B = 5100

Distance = sqrt((5-6)² + (5000-5100)²)
        = sqrt(1 + 10000)
        = sqrt(10001)
        ≈ 100

Contribution of Gene_A: 1 (0.01%)
Contribution of Gene_B: 10000 (99.99%)
```

Gene A's contribution is essentially ignored!

**When normalization is critical:**

1. **Distance-based algorithms**: K-Means, KNN, SVM with RBF kernel
2. **Gradient descent**: Neural networks, logistic regression
3. **Regularized models**: Lasso, Ridge regression
4. **Principal Component Analysis (PCA)**

**When normalization may not be necessary:**

1. **Tree-based methods**: Decision trees, random forests (they split on individual features)
2. **Naive Bayes**
3. **When all features are on the same scale already**

### 2.4.2 Min-Max Scaling (Normalization)

Min-Max scaling transforms features to a fixed range, typically 0 to 1.

**Formula:**
```
scaled_value = (value - min) / (max - min)
```

**Example:**

Original expression values for Gene_1: [2, 5, 8, 11]
- Minimum: 2
- Maximum: 11
- Range: 11 - 2 = 9

Scaled values:
- 2 → (2-2)/9 = 0.0
- 5 → (5-2)/9 = 0.333
- 8 → (8-2)/9 = 0.667
- 11 → (11-2)/9 = 1.0

**Implementation:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['GENE_1', 'GENE_2']] = scaler.fit_transform(df[['GENE_1', 'GENE_2']])
```

**Advantages:**
- Bounded output (always between 0 and 1)
- Preserves zero values if minimum is zero
- Intuitive interpretation

**Disadvantages:**
- Sensitive to outliers (a single extreme value affects the whole scaling)
- Doesn't center the data
- New data might fall outside [0,1] range

**When to use:**
- Neural networks (especially with sigmoid/tanh activations)
- Image data (pixel values)
- When you need bounded values
- Features don't have outliers

**Biological example:**

You're building a neural network to classify cancer subtypes using gene expression. Min-Max scaling ensures all genes contribute equally to the first layer's activation, regardless of their baseline expression levels.

### 2.4.3 Standardization (Z-score Normalization)

Standardization transforms features to have mean=0 and standard deviation=1.

**Formula:**
```
z = (value - mean) / standard_deviation
```

**Example:**

Original expression values for Gene_1: [2, 5, 8, 11]
- Mean: (2+5+8+11)/4 = 6.5
- Standard deviation: 3.696

Standardized values:
- 2 → (2-6.5)/3.696 = -1.217
- 5 → (5-6.5)/3.696 = -0.406
- 8 → (8-6.5)/3.696 = 0.406
- 11 → (11-6.5)/3.696 = 1.217

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['GENE_1', 'GENE_2']] = scaler.fit_transform(df[['GENE_1', 'GENE_2']])
```

**Interpretation of z-scores:**
- z = 0: Value is at the mean
- z = 1: Value is one standard deviation above mean
- z = -2: Value is two standard deviations below mean
- |z| > 3: Potential outlier

**Advantages:**
- Less sensitive to outliers than Min-Max
- Centers the data (mean = 0)
- Works well with algorithms assuming normally distributed data
- Can handle new data outside original range

**Disadvantages:**
- Output is unbounded (can be any value)
- Assumes approximate normal distribution
-
