# Chapter 2: Data Cleaning and Exploratory Data Analysis for Biological Data

## Introduction

In the world of machine learning, there's a popular saying: "garbage in, garbage out." This principle is especially true in biological research, where data quality can make or break your analysis. While the excitement of machine learning often centers on sophisticated algorithms and impressive predictions, experienced practitioners know that the real work, and often the most critical work, happens long before any model is trained.

This chapter focuses on the essential but often overlooked foundation of any successful machine learning project: data preparation and exploratory data analysis (EDA). We will explore how to work with biological datasets, understand Python's core data structures, identify and fix data quality issues, and prepare your data for machine learning algorithms. By the end of this chapter, you will understand why data scientists often spend 80% of their time on data preparation and only 20% on actual modeling.

## 2.1 Python Data Structures: Building Blocks for Biological Data Analysis

Before we dive into data cleaning and analysis, we need to understand how Python stores and organizes data. Think of data structures as containers, each designed for specific purposes, much like how a biologist uses different containers for different specimens.

### 2.1.1 Lists: Your First Data Container

Lists are Python's most basic sequential data structure. Imagine you are cataloging genes of interest from a recent study. A list lets you store these gene names in order:

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

While lists and dictionaries are fundamental, real biological data analysis requires something more powerful: the Pandas DataFrame. If lists are like single-file folders and dictionaries are like filing cabinets, DataFrames are like entire spreadsheet applications.

**What is a DataFrame?**
A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. Think of it as an Excel spreadsheet or a SQL table in Python. It is the primary data structure you will use for biological data analysis.

**Creating a simple DataFrame:**
```python
import pandas as pd

gene_data = pd.DataFrame({
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

Imagine you have performed RNA-seq on 100 patients, 60 healthy and 40 with disease. For each patient, you have measured the expression of 50 genes. Your DataFrame might look like:

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

This type of filtering is fundamental to biological data analysis, identifying genes or samples that meet specific criteria.

## 2.2 Data Quality Issues in Biological Datasets

Real-world biological data is messy. Unlike the clean datasets in textbooks, actual experimental data comes with numerous quality issues. Understanding these problems is the first step in addressing them.

### 2.2.1 Missing Values: The Unavoidable Reality

Missing values are perhaps the most common data quality issue in biological research. They occur for many reasons, and how you handle them can significantly impact your analysis.

**Why do missing values occur?**

1. **Technical failures**: A sample did not amplify in PCR, a microarray spot was defective, or a sequencing read failed quality control.

2. **Below detection limit**: Gene expression too low to be reliably measured. In RNA-seq, genes with very low read counts are often filtered out.

3. **Random chance**: Sometimes instruments fail, reagents degrade, or human error occurs.

4. **Study design**: Not all measurements were taken for all samples (e.g., some clinical variables only measured in disease patients).

**How missing values appear in data:**

Missing values can be represented in many ways:
- Python's `None` or `NaN` (Not a Number)
- Empty cells in spreadsheets
- Special codes like -999, 0, or "NA"
- Literally the text "missing" or "N/A"

This inconsistency is why data cleaning is crucial; you need to standardize how missing values are represented before analysis.

**Types of missingness:**

Understanding why data is missing helps you decide how to handle it:

**1. Missing Completely at Random (MCAR):**

The probability of being missing is the same for all observations. The missingness is unrelated to any observed or unobserved data.

**Example**: A weighing scale that ran out of batteries, some data will be missing simply because of bad luck.

In biological research: A batch of samples was lost due to a freezer malfunction, unrelated to any biological or technical factors.

**2. Missing at Random (MAR):**

The probability of being missing is related to other observed variables but not the missing value itself.

**Example**: When placed on a soft surface, a weighing scale may produce more missing values than when placed on a hard surface.

In biological research: Older patients were less likely to have certain tests performed, but among patients of the same age, missingness is random.

**3. Missing Not at Random (MNAR):**

The fact that the data are missing is systematically related to the unobserved data; the missingness is related to the value that would have been observed.

**Example**: In a depression registry, participants with severe depression are more likely to refuse to complete the survey about depression severity.

In biological research: Gene expression below the detection limit is missing precisely because it is low. This is the most problematic type.

**Example in RNA-seq data:**

RNA-seq commonly has 5-15% missing values. Genes with very low expression might have:
- Sample 1: 0.2 (measured)
- Sample 2: NaN (below detection)
- Sample 3: 0.1 (measured)
- Sample 4: NaN (below detection)

The missingness is MNAR, it is missing because the expression is low, which is exactly what we are trying to measure!

### 2.2.2 Outliers: Signal or Noise?

Outliers are extreme values that differ substantially from other observations. In biology, the challenge is distinguishing between:
- **Technical errors**: Measurement mistakes, contamination, equipment malfunction
- **Biological truth**: Genuine extreme biological variation

**Example scenario:**

You are measuring gene expression across 100 samples:
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
- **Remove**: Only if you are confident it is an error and have documentation

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

Python treats "BRCA1" and "brca1" as completely different strings. If you are filtering for "BRCA1", you will miss any entries recorded as "brca1".

**Solutions:**

1. **Standardize text case**:
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
3. **Correlation analysis**: Samples from the same batch are more similar than expected

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
Remove any row that has any missing values.

```python
df_clean = df.dropna()
```

**Advantages:**
- Simple and straightforward
- No assumptions about missing data
- Preserves observed relationships

**Disadvantages:**
- Can lose substantial data (if any column has missing values, the entire row is removed)
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
- Does not account for correlations

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

Some approaches do not require imputation.

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
- You are using these specific algorithms
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

When you feed this into a machine learning algorithm that calculates distances (like K-Nearest Neighbors or K-Means clustering), Gene B dominates the calculation simply because its numbers are larger, even if Gene A is more biologically relevant for your question!

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
- Does not center the data
- New data might fall outside [0,1] range

**When to use:**
- Neural networks (especially with sigmoid/tanh activations)
- Image data (pixel values)
- When you need bounded values
- Features do not have outliers

**Biological example:**

You are building a neural network to classify cancer subtypes using gene expression. Min-Max scaling ensures all genes contribute equally to the first layer's activation, regardless of their baseline expression levels.

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
- Does not preserve zero values

**When to use:**
- Most machine learning algorithms (logistic regression, SVM, KNN)
- When features are approximately normally distributed
- PCA and other dimensionality reduction techniques
- When you do not need bounded output

**Biological example:**

Gene expression data after log transformation often approximates normal distribution. Standardization puts all genes on the same scale while preserving their relative variation.

### 2.4.4 Log Transformation

Before or instead of scaling, biological data often benefits from log transformation.

**Why log transformation for gene expression?**

1. **Reduces right skew**: Gene expression is typically log-normal (most genes low, few very high)
2. **Stabilizes variance**: Makes variance more constant across expression levels
3. **Makes fold-changes symmetric**: 2-fold up and 2-fold down become equidistant
4. **Approximates normality**: Many statistical tests assume normal distribution

**Implementation:**
```python
# Add 1 to avoid log(0) = undefined
df['GENE_1_log'] = np.log2(df['GENE_1'] + 1)

# Or natural log
df['GENE_1_ln'] = np.log(df['GENE_1'] + 1)
```

**Why add 1?**
- log(0) is undefined
- Adding 1 ensures log(0+1) = log(1) = 0
- Called "pseudocount" in genomics

**Example:**

Original values: [1, 10, 100, 1000]
Log2 transformed: [1, 3.46, 6.66, 9.97]

The 1000-fold range (1 to 1000) becomes a 9-fold range (1 to 9.97), making the data more manageable.

### 2.4.5 Robust Scaling

Robust scaling uses median and interquartile range instead of mean and standard deviation, making it resistant to outliers.

**Formula:**
```
scaled_value = (value - median) / IQR
```

where IQR = 75th percentile - 25th percentile

**Implementation:**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['GENE_1', 'GENE_2']] = scaler.fit_transform(df[['GENE_1', 'GENE_2']])
```

**When to use:**
- Data has outliers
- You want to preserve outlier information (not remove or cap them)
- Median is more appropriate than mean for your data

### 2.4.6 Choosing a Scaling Method

**Decision guide:**

```
Does data have outliers?
├─ Yes → RobustScaler or remove outliers first
└─ No
   │
   Is data approximately normal?
   ├─ Yes → StandardScaler
   └─ No
      │
      Is it log-normal (gene expression, read counts)?
      ├─ Yes → Log transform, then StandardScaler
      └─ Do you need bounded output [0,1]?
         ├─ Yes → MinMaxScaler
         └─ No → StandardScaler (default choice)
```

**Algorithm-specific recommendations:**

- **Neural Networks**: MinMaxScaler or StandardScaler
- **SVM**: StandardScaler
- **K-Means**: StandardScaler
- **Linear/Logistic Regression**: StandardScaler
- **Tree-based (Random Forest, XGBoost)**: Often not necessary
- **PCA**: StandardScaler (almost always)

## 2.5 Basic Statistics for Biological Data

Before machine learning, we need to understand our data through basic statistics. This is the "exploration" in Exploratory Data Analysis.

### 2.5.1 Measures of Central Tendency

**Mean (Average):**

The sum divided by count. Most common measure but sensitive to outliers.

```python
mean_expression = df['GENE_1'].mean()
```

**Example:**
Gene expression values: [5, 6, 5, 7, 100]
Mean = (5+6+5+7+100)/5 = 24.6

The outlier (100) pulls the mean far from typical values.

**Median (Middle value):**

The middle value when data is sorted. Robust to outliers.

```python
median_expression = df['GENE_1'].median()
```

**Same example:**
Sorted: [5, 5, 6, 7, 100]
Median = 6 (the middle value)

Much more representative of typical expression!

**Mode (Most common):**

Most frequent value. Mainly used for categorical data.

```python
most_common_condition = df['Condition'].mode()[0]
```

**When to use each:**
- **Mean**: Normal distribution, no outliers, want to use all information
- **Median**: Skewed data, outliers present, more robust estimate
- **Mode**: Categorical data, want most common category

**Biological insight:**

For gene expression, median is often preferred because:
- Distribution is typically skewed
- Outliers (highly expressed genes) are common
- More robust to technical artifacts

### 2.5.2 Measures of Spread

**Range:**

Difference between maximum and minimum.

```python
expression_range = df['GENE_1'].max() - df['GENE_1'].min()
```

Simple but heavily influenced by outliers.

**Variance:**

Average squared deviation from mean.

```python
variance = df['GENE_1'].var()
```

**Formula:**
```
variance = Σ(xi - mean)² / (n-1)
```

Large variance means data is spread out; small variance means data is clustered.

**Standard Deviation (SD):**

Square root of variance. In the same units as original data.

```python
std_dev = df['GENE_1'].std()
```

**Interpretation:**
For approximately normal data, roughly:
- 68% of values within mean ± 1 SD
- 95% of values within mean ± 2 SD
- 99.7% of values within mean ± 3 SD

**Example:**

Gene expression with mean=10, SD=2:
- About 68% of samples have expression between 8 and 12
- About 95% have expression between 6 and 14
- Values above 16 or below 4 are unusual (potential outliers)

**Coefficient of Variation (CV):**

Standard deviation divided by mean, expressed as percentage.

```python
cv = (df['GENE_1'].std() / df['GENE_1'].mean()) * 100
```

Useful for comparing variability of genes with different expression levels.

**Example:**
- Gene A: Mean=100, SD=10, CV=10%
- Gene B: Mean=10, SD=1, CV=10%

Both have the same relative variability despite different absolute values.

**Biological application:**

In quality control, you might calculate CV for each gene across replicates. High CV (>30%) might indicate:
- Technical problems (poor measurement reproducibility)
- High biological variability
- Presence of outliers

### 2.5.3 Quantiles and Percentiles

Percentiles divide data into 100 equal parts. Quartiles divide into 4 parts.

**Quartiles:**
- Q1 (25th percentile): 25% of data is below this value
- Q2 (50th percentile): The median
- Q3 (75th percentile): 75% of data is below this value

**Interquartile Range (IQR):**
```python
Q1 = df['GENE_1'].quantile(0.25)
Q3 = df['GENE_1'].quantile(0.75)
IQR = Q3 - Q1
```

IQR contains the middle 50% of data. Used to identify outliers:
- Lower outliers: < Q1 - 1.5 × IQR
- Upper outliers: > Q3 + 1.5 × IQR

**Biological example:**

Expression values: [2, 3, 3, 4, 5, 5, 6, 7, 8, 20]
- Q1 = 3
- Q3 = 7
- IQR = 4
- Upper fence = 7 + 1.5×4 = 13
- Value 20 is an outlier (> 13)

### 2.5.4 Correlation Analysis

Correlation measures the strength and direction of linear relationships between variables.

**Pearson Correlation Coefficient (r):**

Measures linear correlation. Range: -1 to +1.

```python
correlation = df['GENE_1'].corr(df['GENE_2'])
```

**Interpretation:**
- r = 1: Perfect positive correlation (as one increases, other increases proportionally)
- r = 0: No linear correlation
- r = -1: Perfect negative correlation (as one increases, other decreases)
- |r| > 0.7: Strong correlation
- 0.3 < |r| < 0.7: Moderate correlation
- |r| < 0.3: Weak correlation

**Biological examples:**

**High positive correlation (r=0.95):**
Two technical replicates of the same sample should be highly correlated. If not, there's a technical problem!

**Moderate positive correlation (r=0.6):**
Genes in the same pathway often show moderate correlation. They are co-regulated but also have independent regulation.

**Negative correlation (r=-0.7):**
Tumor suppressor and oncogene might show negative correlation. When one is up, the other tends to be down.

**Correlation matrix:**

For multiple genes, create a correlation matrix:

```python
correlation_matrix = df[['GENE_1', 'GENE_2', 'GENE_3', 'GENE_4']].corr()
```

This creates a symmetric matrix where element (i,j) is the correlation between Gene i and Gene j.

**Visualization:**

Heatmaps are perfect for correlation matrices:

```python
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
```

Colors show correlation strength:
- Red: Strong positive
- Blue: Strong negative
- White: No correlation

**Important caveat:**

Correlation only measures linear relationships! Genes might have complex, non-linear relationships that correlation does not capture.

**Example:**

Gene A expression: [1, 2, 3, 4, 5]
Gene B expression: [1, 4, 9, 16, 25]  # B = A²

These have a perfect mathematical relationship, but Pearson correlation is not 1 because the relationship is non-linear!

### 2.5.5 Group Comparisons

Often you want to compare groups (disease vs. healthy, treatment vs. control).

**Group statistics:**

```python
# Mean expression by condition
group_means = df.groupby('Condition')['GENE_1'].mean()

# Multiple statistics
group_stats = df.groupby('Condition')['GENE_1'].agg(['mean', 'std', 'count'])
```

**Example output:**
```
Condition
Healthy    5.2 ± 1.3 (n=60)
Disease    8.9 ± 2.1 (n=40)
```

**Interpretation:**

Disease samples have higher Gene_1 expression (8.9 vs 5.2) with more variability (SD 2.1 vs 1.3).

**Questions to ask:**
1. Is the difference meaningful biologically?
2. Is it statistically significant?
3. Is the variance similar between groups?
4. Are there outliers affecting the means?

**Comprehensive summary:**

```python
summary = df.describe()
```

This gives:
- count (sample size)
- mean
- std (standard deviation)
- min
- 25% (first quartile)
- 50% (median)
- 75% (third quartile)
- max

All the basic statistics in one convenient table!

## 2.6 Exploratory Data Analysis (EDA) Workflow

EDA is a systematic approach to understanding your data before modeling. Think of it as the reconnaissance mission before the battle.

### 2.6.1 Step 1: Load and Inspect

**Initial data loading:**

```python
import pandas as pd
df = pd.read_csv('gene_expression.csv')
```

**First look:**

```python
# First few rows
print(df.head())

# Last few rows (check if data is complete)
print(df.tail())

# Random sample (get sense of variety)
print(df.sample(5))
```

**Structure check:**

```python
# Dimensions
print(f"Shape: {df.shape}")  # (100 samples, 54 features)

# Column names and types
print(df.info())

# Data types
print(df.dtypes)
```

**What to look for:**
- Are numeric columns actually numeric? (Sometimes read as text)
- Are there unexpected columns?
- Is the data size what you expected?
- Are column names clean and meaningful?

**Example issue:**

```
Sample_ID     object  # Should be fine
Expression   object  # PROBLEM! Should be float64
```

If Expression is read as object (text), it might contain non-numeric values or be formatted incorrectly.

**Fix:**
```python
df['Expression'] = pd.to_numeric(df['Expression'], errors='coerce')
```

This converts to numeric, turning invalid values to NaN.

### 2.6.2 Step 2: Clean Data

**Check for duplicates:**

```python
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove exact duplicates
df = df.drop_duplicates()

# Check for duplicate sample IDs
duplicate_ids = df['Sample_ID'].duplicated().sum()
print(f"Duplicate Sample IDs: {duplicate_ids}")
```

**Assess missing values:**

```python
# Missing count per column
missing = df.isnull().sum()
print(missing[missing > 0])

# Missing percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])
```

**Decide on handling:**

```python
# Drop columns with >20% missing
threshold = len(df) * 0.2
df = df.dropna(thresh=len(df) - threshold, axis=1)

# Impute remaining with median
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col].fillna(df[col].median(), inplace=True)
```

**Fix data types:**

```python
# Convert categorical to category type
df['Condition'] = df['Condition'].astype('category')
df['Gender'] = df['Gender'].astype('category')

# Ensure gene expression is numeric
gene_columns = [col for col in df.columns if col.startswith('GENE_')]
df[gene_columns] = df[gene_columns].apply(pd.to_numeric, errors='coerce')
```

**Standardize formatting:**

```python
# Standardize text to uppercase
df['Sample_ID'] = df['Sample_ID'].str.upper()

# Remove whitespace
df['Condition'] = df['Condition'].str.strip()

# Replace values
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
```

### 2.6.3 Step 3: Explore with Statistics

**Overall summary:**

```python
# Numeric variables summary
print(df.describe())

# Include categorical variables
print(df.describe(include='all'))
```

**By-group analysis:**

```python
# Compare conditions
condition_comparison = df.groupby('Condition').agg({
    'Age': ['mean', 'std', 'min', 'max'],
    'GENE_1': ['mean', 'std'],
    'GENE_2': ['mean', 'std']
})
print(condition_comparison)
```

**Correlation analysis:**

```python
# Correlation matrix for all genes
gene_corr = df[gene_columns].corr()

# Find highly correlated gene pairs
high_corr = []
for i in range(len(gene_corr.columns)):
    for j in range(i+1, len(gene_corr.columns)):
        if abs(gene_corr.iloc[i, j]) > 0.8:
            high_corr.append({
                'Gene1': gene_corr.columns[i],
                'Gene2': gene_corr.columns[j],
                'Correlation': gene_corr.iloc[i, j]
            })

high_corr_df = pd.DataFrame(high_corr)
print(high_corr_df)
```

### 2.6.4 Step 4: Visualize

Visualization reveals patterns invisible in tables of numbers.

**Distribution plots:**

Histograms show the shape of data distribution.

```python
import matplotlib.pyplot as plt

plt.hist(df['GENE_1'], bins=30, edgecolor='black')
plt.xlabel('Expression Level')
plt.ylabel('Frequency')
plt.title('Distribution of GENE_1 Expression')
plt.show()
```

**What to look for:**
- **Bell curve**: Normal distribution
- **Right skew**: Most values low, few very high (typical for gene expression)
- **Left skew**: Most values high, few very low
- **Bimodal**: Two peaks (might indicate subgroups!)
- **Outliers**: Isolated bars far from main distribution

**Box plots for group comparison:**

```python
df.boxplot(column='GENE_1', by='Condition')
plt.ylabel('Expression Level')
plt.title('GENE_1 Expression by Condition')
plt.show()
```

**What box plots show:**
- **Box**: Contains middle 50% of data (IQR)
- **Line in box**: Median
- **Whiskers**: Extend to 1.5 × IQR
- **Dots beyond whiskers**: Outliers

**Interpretation example:**

```
Healthy:  Median=5, IQR=4-6, no outliers
Disease:  Median=9, IQR=7-11, one outlier at 20
```

Clear separation suggests GENE_1 could be a biomarker!

**Scatter plots for relationships:**

```python
plt.scatter(df['GENE_1'], df['GENE_2'], 
           c=df['Condition'].map({'Healthy': 'blue', 'Disease':  ared'}),
           alpha=0.6)
plt.xlabel('GENE_1 Expression')
plt.ylabel('GENE_2 Expression')
plt.title('GENE_1 vs GENE_2')
plt.legend(['Healthy', 'Disease'])
plt.show()
```

**What to look for:**
- **Linear pattern**: Strong correlation
- **Clustering**: Distinct groups
- **Color separation**: Groups differ in both genes
- **Outliers**: Points far from clusters

**Correlation heatmap:**

```python
import seaborn as sns

correlation_matrix = df[gene_columns[:10]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
           center=0, vmin=-1, vmax=1)
plt.title('Gene Expression Correlations')
plt.show()
```

**Interpretation:**
- **Blocks of red**: Genes co-expressed (might be in same pathway)
- **Blue clusters**: Inversely expressed
- **Diagonal red**: Perfect self-correlation (always 1)

### 2.6.5 Step 5: Feature Engineering

Sometimes you need to create new features from existing ones.

**Log transformation for skewed data:**

```python
# Log transform gene expression
df['GENE_1_log'] = np.log2(df['GENE_1'] + 1)
```

**Creating categorical bins:**

```python
# Age groups
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[0, 40, 60, 100],
                         labels=['Young', 'Middle', 'Senior'])
```

**Ratios and interactions:**

```python
# Gene expression ratio
df['GENE_Ratio'] = df['GENE_1'] / (df['GENE_2'] + 0.001)  # Add small value to avoid division by zero

# Interaction terms
df['Age_x_GENE1'] = df['Age'] * df['GENE_1']
```

**Creating summary features:**

```python
# Average expression across gene family
gene_family = ['GENE_1', 'GENE_2', 'GENE_3']
df['Family_Mean'] = df[gene_family].mean(axis=1)
```

### 2.6.6 Step 6: Prepare for Modeling

**Separate features and target:**

```python
# Define what you are predicting
target = 'Condition'

# Define what you are using to predict
feature_columns = gene_columns + ['Age', 'Gender']

X = df[feature_columns]
y = df[target]
```

**Encode categorical variables:**

Machine learning algorithms need numbers, not text.

```python
from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Or for features, use one-hot encoding
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
```

**Now ready for train/test split and scaling (covered in next section)!**

## 2.7 Train/Test Split: The Foundation of Honest Evaluation

The train/test split is one of the most important concepts in machine learning. Get this wrong, and all your results are meaningless.

### 2.7.1 Why Split Data?

**The fundamental problem:**

If you train your model on data and test it on the same data, you cannot know if the model learned general patterns or just memorized the specific examples.

**Analogy:**

Imagine teaching students by giving them practice problems. Then, on the exam, you give them the exact same problems. They will ace it! But have they actually learned mathematics, or did they just memorize solutions?

To truly assess learning, you must test on new, unseen problems.

**The same principle applies to machine learning:**

- **Training data**: The "practice problems" the model learns from
- **Test data**: The "exam" that evaluates how well it learned
- The test data must be completely separate, the model never sees it during training

### 2.7.2 Implementing Train/Test Split

**Basic split:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42      # For reproducibility
)
```

This creates four datasets:
- `X_train`: Features for training (80%)
- `X_test`: Features for testing (20%)
- `y_train`: Target for training (80%)
- `y_test`: Target for testing (20%)

**Choosing test size:**

- **Large datasets (>10,000 samples)**: 10-20% test is fine
- **Medium datasets (1,000-10,000)**: 20-30% test
- **Small datasets (<1,000)**: 20-30% test, consider cross-validation

Trade-off:
- Larger training set → Model learns better
- Larger test set → More reliable performance estimate

### 2.7.3 Random State: Ensuring Reproducibility

The `random_state` parameter is crucial for reproducible research.

**Without random_state:**

```python
# Run 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)
# Accuracy: 0.87

# Run 2 (different random split)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)
# Accuracy: 0.91
```

Different splits give different results! Which one do you report?

**With random_state:**

```python
# Run 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
# Accuracy: 0.89

# Run 2 (same split)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
# Accuracy: 0.89
```

Same split every time! Results are reproducible. You and your collaborators get identical results.

**What number to use?**
Does not matter! Common choices: 42, 0, 123. Just pick one and stick with it.

### 2.7.4 Stratified Split: Maintaining Class Balance

**The problem:**

Imagine you have 100 samples: 90 healthy, 10 diseased.

A random split might give you:
- Training: 88 healthy, 2 disease (2% disease)
- Testing: 2 healthy, 8 disease (80% disease!)

The test set is not representative!

**Solution: Stratified splitting**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class proportions
)
```

Now you get:
- Training: 72 healthy, 8 disease (10% disease)
- Testing: 18 healthy, 2 disease (10% disease)

Both sets have the same 90:10 ratio as the original data.

**When to use stratification:**
- Classification problems (almost always)
- Imbalanced datasets (absolutely essential)
- You want training and test sets to be similarly distributed

### 2.7.5 Data Leakage: The Cardinal Sin

Data leakage occurs when information from the test set influences the training process. This is one of the worst mistakes in machine learning because it makes results artificially good but worthless for real-world deployment.

**Common leakage mistake: Scaling before splitting**

**WRONG:**
```python
# Scale entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then split
X_train, X_test = train_test_split(X_scaled, ...)
```

**Why this is wrong:**

When you fit the scaler on the entire dataset, it calculates mean and standard deviation using ALL data, including the test set. The training process has "seen" information about the test set (its mean and standard deviation).

**Example of the problem:**

Original test set Gene_1 values: [15, 18, 20]
Overall mean: 10, std: 5

Scaling using overall statistics:
- 15 becomes (15-10)/5 = 1.0
- The scaler "knew" that values around 15-20 exist (from seeing the test set)

If test values were completely different (e.g., 100, 105, 110), the overall mean and std would be different, leading to different scaled values.

**CORRECT:**
```python
# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then scale, fitting ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training
X_test_scaled = scaler.transform(X_test)  # Only transform test (using training statistics)
```

**Key points:**
- `fit_transform()` on training: Calculate mean/std from training, then scale
- `transform()` on test: Use training's mean/std to scale test

This ensures test data truly remains unseen!

**Other sources of data leakage:**

1. **Feature selection using entire dataset:**

   **Wrong:**
   ```python
   selected_features = select_features(X, y)  # Uses all data
   X_selected = X[selected_features]
   X_train, X_test = train_test_split(X_selected, y)
   ```

   **Correct:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   selected_features = select_features(X_train, y_train)  # Only training
   X_train_selected = X_train[selected_features]
   X_test_selected = X_test[selected_features]
   ```

2. **Using test set to tune hyperparameters:**

   Never adjust your model based on test set performance! Use cross-validation on training set for hyperparameter tuning.

3. **Duplicate samples in both training and test:**

   If the same sample appears in both sets, the model has "seen" the test data!

### 2.7.6 Complete Preparation Pipeline

Putting it all together in the correct order:

```python
# 1. Clean data (whole dataset)
df = handle_missing_values(df)
df = remove_duplicates(df)

# 2. Separate features and target
X = df[feature_columns]
y = df[target_column]

# 3. Encode categorical target if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. SPLIT FIRST!
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 5. Handle categorical features (fit on training only)
encoder = pd.get_dummies(X_train, columns=['Gender'])
X_test = pd.get_dummies(X_test, columns=['Gender'])

# Ensure both have same columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 6. Scale (fit on training, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Now ready for machine learning!
```

**Critical order:**
1. Clean data
2. Separate X and y
3. **SPLIT**
4. Feature engineering (fit on training)
5. Scaling (fit on training)
6. Model training and evaluation

## 2.8 Practical Biological Considerations

### 2.8.1 High-Dimensional Data (p >> n)

**The problem:**

In genomics, we often have:
- n = number of samples (often 50-200)
- p = number of features (often 20,000+ genes)

When p >> n (more features than samples), special challenges arise:
- **Overfitting risk**: Model can perfectly fit training data by chance
- **Curse of dimensionality**: Data becomes sparse in high dimensions
- **Computational challenges**: Many algorithms struggle

**Solutions:**

1. **Feature selection**: Choose most relevant genes
   - Differential expression analysis
   - Variance filtering (remove low-variance genes)
   - Correlation filtering (remove redundant genes)

2. **Dimensionality reduction**: 
   - PCA (Principal Component Analysis)
   - t-SNE for visualization
   - Autoencoders

3. **Regularization**: 
   - Lasso (L1): Automatically selects features
   - Ridge (L2): Shrinks coefficients
   - Elastic Net: Combination

4. **Specialized algorithms**:
   - Random Forests (handle many features well)
   - XGBoost (built-in feature selection)

### 2.8.2 Batch Effects

**Detecting batch effects:**

```python
# PCA colored by batch
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=batch_labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA colored by batch')
plt.show()
```

If samples cluster by batch rather than by biology, you have a batch effect problem!

**Correcting batch effects:**

```python
# Using ComBat (pyComBat package)
from combat.pycombat import pycombat

corrected_data = pycombat(data, batch)
```

**Best practice:**

Design experiments to balance batches:
- Include healthy and disease in each batch
- Randomize sample processing order
- Document all batch information

### 2.8.3 Class Imbalance

**The problem:**

Disease datasets often have:
- 90% healthy (controls)
- 10% disease (cases)

A naive classifier that always predicts "healthy" gets 90% accuracy but is useless!

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

4. **Use appropriate metrics**: Do not rely on accuracy alone
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

1. **Do not randomize time points**: Keep temporal order intact
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

**Why it is wrong:**

You have now used the test set to make modeling decisions! It is no longer truly unseen. Your reported 85% is optimistic.

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

Six months later, you cannot remember exactly what you did. Reviewers ask questions you cannot answer.

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

If you impute missing values with mean=12.7, you are filling in values much higher than most of the data!

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
print(df['Age'].mean())  # Error: cannot calculate mean of object type!
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

Do not forget to properly encode categorical variables before modeling:

```python
# Wrong: treating "Tissue_Type" as if it were numeric
X = df[['Tissue_Type', 'Gene_1', 'Gene_2']]  # Tissue_Type is text!

# Correct: one-hot encode
X = pd.get_dummies(df[['Tissue_Type', 'Gene_1', 'Gene_2']], 
                   columns=['Tissue_Type'])
```

## 2.10 Complete Real-World Example: Breast Cancer RNA-seq Analysis

Let's walk through a complete, realistic example from start to finish, applying everything we have learned.

### 2.10.1 The Scenario

You have received RNA-seq data from a collaborator studying breast cancer. The dataset contains:
- 120 patient samples (80 cancer, 40 healthy)
- 500 genes selected as potentially relevant
- Patient metadata: age, tumor stage, treatment status

Your goal: Build a classifier to distinguish cancer from healthy tissue based on gene expression.

### 2.10.2 Complete Analysis Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and inspect
df = pd.read_csv('breast_cancer_expression.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['Diagnosis'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum().sum()}")

# Step 2: Data cleaning
# Remove genes with >20% missing
gene_columns = [col for col in df.columns if col.startswith('GENE_')]
missing_per_gene = df[gene_columns].isnull().sum() / len(df)
genes_to_drop = missing_per_gene[missing_per_gene > 0.2].index
df = df.drop(columns=genes_to_drop)
print(f"Dropped {len(genes_to_drop)} genes with >20% missing")

# Impute remaining with median
gene_columns = [col for col in df.columns if col.startswith('GENE_')]
for gene in gene_columns:
    if df[gene].isnull().sum() > 0:
        df[gene].fillna(df[gene].median(), inplace=True)

# Handle metadata missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Step 3: Exploratory analysis
print("\nAge by Diagnosis:")
print(df.groupby('Diagnosis')['Age'].describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.boxplot(column='Age', by='Diagnosis', ax=axes[0])
axes[0].set_title('Age Distribution')
df.boxplot(column='GENE_1', by='Diagnosis', ax=axes[1])
axes[1].set_title('GENE_1 Expression')
plt.tight_layout()
plt.savefig('eda_plots.png')

# Step 4: Feature engineering
# Log-transform skewed genes
skewed_genes = []
for gene in gene_columns:
    if abs(df[gene].skew()) > 1:
        df[f'{gene}_log'] = np.log2(df[gene] + 1)
        skewed_genes.append(gene)

print(f"Log-transformed {len(skewed_genes)} skewed genes")

# Step 5: Prepare for modeling
log_features = [f'{gene}_log' for gene in skewed_genes]
feature_columns = log_features + ['Age']

X = df[feature_columns]
y = df['Diagnosis']

# Encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 6: Train-test split (BEFORE scaling!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance: {np.bincount(y_test)}")

# Step 7: Scale features (fit on training only!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaling verification:")
print(f"Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

# Save processed data
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\n✓ Data preparation complete!")
print("Ready for machine learning modeling!")
```

### 2.10.3 Key Takeaways from the Example

**Critical steps we followed:**
1. ✓ Loaded and inspected data structure
2. ✓ Cleaned missing values systematically
3. ✓ Performed exploratory analysis
4. ✓ Applied feature engineering (log transformation)
5. ✓ Split data BEFORE scaling
6. ✓ Fit scaler on training data only
7. ✓ Verified no data leakage
8. ✓ Documented each step

**Results:**
- Original: 120 samples, 500 genes
- After cleaning: 120 samples, 498 genes
- After feature engineering: 350 features (log-transformed + age)
- Train: 96 samples (64 cancer, 32 healthy)
- Test: 24 samples (16 cancer, 8 healthy)
- Class balance maintained: 2:1 ratio preserved

## 2.11 Summary and Key Takeaways

This chapter covered the essential foundations of data preparation for machine learning in biology:

**Core Concepts:**

1. **Python data structures**: Lists for sequences, dictionaries for key-value pairs, DataFrames for tabular data

2. **Data quality issues**: Missing values, outliers, inconsistent formatting, duplicates, and batch effects are the norm in biological data

3. **Missing value strategies**: Choose between deletion, imputation, or modeling based on amount and type of missingness (MCAR, MAR, MNAR)

4. **Normalization and scaling**: Essential for most ML algorithms; choose Min-Max for bounded output, Standardization for most cases, log transformation for skewed biological data

5. **Basic statistics**: Use mean, median, standard deviation, and correlation to understand your data before modeling

6. **EDA workflow**: Load → Clean → Explore → Visualize → Engineer → Prepare

7. **Train/test split**: The foundation of honest evaluation; always split before scaling to avoid data leakage

**Biological Data Challenges:**

- High dimensionality (p >> n)
- Batch effects from multi-center studies
- Class imbalance in disease datasets
- Non-normal distributions requiring transformation
- Multi-omics integration challenges

**Critical Rules to Remember:**

1. **Never look at test data during development** - it is sacred
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
- Ignoring data types when reading files

**Next Steps:**

With clean, properly prepared data, you are now ready to:
- Apply machine learning algorithms (Chapter 3)
- Evaluate model performance with appropriate metrics
- Interpret results in biological context
- Deploy models for real-world predictions

**Remember:** In machine learning, "garbage in, garbage out" is not just a saying, it is a fundamental truth. The time you invest in proper data preparation will pay dividends in model performance and scientific validity. Good data preparation is not glamorous, but it is the foundation upon which all successful machine learning projects are built.

## 2.12 Further Reading and Resources

**Papers on Missing Data:**
- Little, R. J., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data, 3rd Edition*. Wiley.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data, 2nd Edition*. CRC Press.

**Papers on Batch Effects:**
- Leek, J. T., et al. (2010). Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics*, 11(10), 733-739.
- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.

**Normalization and Scaling:**
- Bolstad, B. M., et al. (2003). A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics*, 19(2), 185-193.
- Evans, C., Hardin, J., & Stoebel, D. M. (2018). Selecting between-sample RNA-Seq normalization methods from the perspective of their assumptions. *Briefings in Bioinformatics*, 19(5), 776-792.

**Online Resources:**
- **Pandas documentation**: https://pandas.pydata.org/docs/
  - Comprehensive guide to DataFrame operations
  - Tutorials on data cleaning and manipulation

- **Scikit-learn preprocessing guide**: https://scikit-learn.org/stable/modules/preprocessing.html
  - Official documentation on scaling, normalization, and encoding

- **Seaborn tutorial**: https://seaborn.pydata.org/tutorial.html
  - Visualization library perfect for statistical graphics

- **Python for Data Analysis (GitHub)**: https://github.com/wesm/pydata-book
  - Code examples from Wes McKinney's book

**Practice Datasets for Biology:**
- **Gene Expression Omnibus (GEO)**: https://www.ncbi.nlm.nih.gov/geo/
  - Thousands of gene expression datasets
  - Well-documented experiments
  
- **The Cancer Genome Atlas (TCGA)**: https://portal.gdc.cancer.gov/
  - Comprehensive cancer genomics data
  - Multi-omics datasets
  
- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/ml/
  - Breast Cancer Wisconsin dataset
  - Various biological classification problems

- **Kaggle Datasets**: https://www.kaggle.com/datasets
  - Gene expression cancer RNA-seq
  - Medical imaging datasets
  - Protein structure datasets

**Software Tools:**
- **ComBat** (batch correction): https://github.com/brentp/combat.py
- **scikit-learn**: https://scikit-learn.org/
- **imbalanced-learn**: https://imbalanced-learn.org/
- **missingno** (visualize missing data): https://github.com/ResidentMario/missingno

**Video Tutorials:**
- StatQuest with Josh Starmer (YouTube): Excellent explanations of statistical concepts
- Corey Schafer's Pandas tutorials (YouTube): Comprehensive pandas programming
- Data School (YouTube): Practical pandas and machine learning

**Communities and Forums:**
- **Biostars**: https://www.biostars.org/ - Bioinformatics Q&A
- **Stack Overflow**: For programming questions
- **Cross Validated**: For statistical questions
- **Reddit r/bioinformatics**: Community discussions

---

**End of Chapter 2**

In the next chapter, we will use this cleaned and prepared data to build our first machine learning models, starting with logistic regression for classification of disease states based on gene expression profiles. We will learn how to train models, make predictions, and evaluate their performance using appropriate metrics for biological applications.
