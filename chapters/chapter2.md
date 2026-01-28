# Chapter 2: Data Cleaning and Exploratory Data Analysis for Biological Data

## Introduction

In the world of machine learning, there's a popular saying: "garbage in, garbage out." This principle is especially true in biological research, where data quality can make or break your analysis. While the excitement of machine learning often centers on sophisticated algorithms and impressive predictions, experienced practitioners know that the real work, and often the most critical work, happens long before any model is trained.

This chapter introduces two Python data structures you will use all the time: **lists** and **dictionaries**. Think of them as the building blocks for storing gene names, sample IDs, expression values, and metadata. Once you are comfortable with these, we will move on to **Pandas** and data cleaning in Chapter 3.

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

## 2.2 Summary and Next Steps

You now know Python’s two main data structures we use before moving to tables:

- **Lists**: ordered sequences of items (e.g. gene names, sample IDs). Use when order matters and you access by position.
- **Dictionaries**: key–value pairs (e.g. gene → expression, sample → metadata). Use when you want to look up by name instead of position.

In **Chapter 3**, we will use **Pandas DataFrames** and **indexing** to load biological data, spot quality issues, handle missing values, normalize and scale, run a simple **EDA workflow**, and introduce the **train/test split**. All of that builds on the lists and dictionaries you have just learned.

---

**End of Chapter 2**

