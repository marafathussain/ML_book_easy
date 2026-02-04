"""
Save the Iris dataset as CSV for use in the book.
Run from ML_book_easy/data/ or project root. Creates iris.csv.
"""
import pandas as pd
from sklearn.datasets import load_iris
import os

base = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(base, "iris.csv")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Use clearer column names (optional: keep sklearn names for consistency with text)
df.columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "species"]
df.to_csv(out_path, index=False)
print(f"Saved {out_path} with shape {df.shape}")
