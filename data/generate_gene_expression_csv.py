"""
Generate dummy gene_expression.csv for Chapter 3 (Data Cleaning & EDA).
Matches columns and examples used in the book. Includes some missing values
for imputation practice. Run once, then you can delete this script.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 60

# Metadata
sample_ids = [f"SAMPLE_{i:03d}" for i in range(1, n + 1)]
conditions = ["Healthy"] * 40 + ["Disease"] * 20
np.random.shuffle(conditions)
ages = np.random.randint(28, 73, size=n).astype(float)
genders = np.random.choice(["M", "F"], size=n)
tissues = np.random.choice(["Breast", "Lung", "Liver", "Colon", "Kidney"], size=n)

# Gene expression (log-normal-ish, 0.5–25)
gene_cols = [f"GENE_{i}" for i in range(1, 11)]
genes = np.exp(np.random.randn(n, 10) * 0.8 + 2).clip(0.5, 25)

# Introduce ~5–8% missing: GENE_1, Age, Tissue
for i in range(n):
    if np.random.rand() < 0.06:
        genes[i, 0] = np.nan
    if np.random.rand() < 0.05:
        ages[i] = np.nan

df = pd.DataFrame(
    {
        "Sample_ID": sample_ids,
        "Condition": conditions,
        "Age": ages,
        "Gender": genders,
        "Tissue": tissues,
    }
)
for j, col in enumerate(gene_cols):
    df[col] = np.round(genes[:, j], 2)

# Add missing Tissue (after DataFrame is built)
miss = np.random.rand(n) < 0.05
df.loc[miss, "Tissue"] = np.nan

out = "gene_expression.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} ({n} rows, {len(df.columns)} columns)")
print(f"Missing: GENE_1={df['GENE_1'].isna().sum()}, Age={df['Age'].isna().sum()}, Tissue={df['Tissue'].isna().sum()}")
