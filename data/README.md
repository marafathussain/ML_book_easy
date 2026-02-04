# Data for ML Book (Easy)

Datasets for hands-on practice with **Chapter 2** and **Chapter 3**.

## Iris dataset (main example)

We use the **Iris flower dataset** throughout the book: 150 flowers, 3 species (setosa, versicolor, virginica), and 4 measurements (sepal length, sepal width, petal length, petal width in cm). It is easy to understand and works well for EDA and first ML models.

### How to load Iris (no file needed)

In Google Colab or any Python environment with scikit-learn:

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Optional: clearer column names
df.columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'species']
```

### Optional: use a CSV file

If you prefer a file (e.g. for upload in Colab), generate **`iris.csv`**:

```bash
cd ML_book_easy/data
python generate_iris_csv.py
```

Then load it:

```python
import pandas as pd
df = pd.read_csv('iris.csv')
```

From the repo root: `df = pd.read_csv('ML_book_easy/data/iris.csv')`.

---

## `gene_expression.csv` (optional)

- **Rows:** 60 samples (patients)
- **Columns:** `Sample_ID`, `Condition` (Healthy/Disease), `Age`, `Gender`, `Tissue`, `GENE_1` … `GENE_10`
- **Missing values:** A few in `GENE_1`, `Age`, and `Tissue` for imputation practice.
- **Use:** Optional alternative for EDA and train/test split if you want a biology-style example.

### Regenerating gene_expression.csv

```bash
cd ML_book_easy/data
python generate_gene_expression_csv.py
```
