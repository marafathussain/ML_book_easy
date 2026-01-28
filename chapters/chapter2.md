# Chapter 2: Data Cleaning and Exploratory Data Analysis for Biological Data

## Introduction

In the world of machine learning, there is a popular saying: "garbage in, garbage out." This principle is especially true in biological research, where data quality can make or break your analysis. While the excitement of machine learning often centers on sophisticated algorithms and impressive predictions, experienced practitioners know that the real work, and often the most critical work, happens long before any model is trained.

This chapter introduces two Python data structures you will use all the time: **lists** and **dictionaries**. Think of them as the building blocks for storing gene names, sample IDs, expression values, and metadata. Once you are comfortable with these, we will move on to **Pandas** and data cleaning in Chapter 3.

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

## 2.2 Summary and Next Steps

You now know Python’s two main data structures we use before moving to tables:

- **Lists**: ordered sequences of items (e.g. gene names, sample IDs). Use when order matters and you access by position.
- **Dictionaries**: key-value pairs (e.g. gene to expression, sample to metadata). Use when you want to look up by name instead of position.

In **Chapter 3**, we will use **Pandas DataFrames** and **indexing** to load biological data, spot quality issues, handle missing values, normalize and scale, run a simple **EDA workflow**, and introduce the **train/test split**. All of that builds on the lists and dictionaries you have just learned.

---

**End of Chapter 2**

