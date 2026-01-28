# Data for ML Book (Easy)

Dummy datasets for hands-on practice with **Chapter 3: Data Cleaning and Exploratory Data Analysis**.

## `gene_expression.csv`

- **Rows:** 60 samples (patients)
- **Columns:** `Sample_ID`, `Condition` (Healthy/Disease), `Age`, `Gender`, `Tissue`, `GENE_1` … `GENE_10`
- **Missing values:** A few in `GENE_1`, `Age`, and `Tissue` for imputation practice.
- **Use:** Load in Pandas, run EDA (summaries, histograms, boxplots, correlation heatmap), practice handling missing values, and try a train/test split.

### How to use

**Option 1 – Same folder as your notebook**

Download `gene_expression.csv` and put it in the same folder as your Jupyter/Colab notebook. Then:

```python
import pandas as pd
df = pd.read_csv('gene_expression.csv')
```

**Option 2 – From the repo (clone or GitHub raw URL)**

If you use the course repo, the file lives in `ML_book_easy/data/`. From the repo root:

```python
df = pd.read_csv('ML_book_easy/data/gene_expression.csv')
```

**Option 3 – Google Colab from GitHub raw URL**

Replace `YOUR_USERNAME` and `YOUR_REPO` with the actual GitHub user and repo name:

```python
url = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/ML_book_easy/data/gene_expression.csv'
df = pd.read_csv(url)
```

### Regenerating the file

To recreate `gene_expression.csv` (e.g. after editing the generator):

```bash
cd ML_book_easy/data
python generate_gene_expression_csv.py
```

You can delete `generate_gene_expression_csv.py` after generating; the CSV is what students need.
