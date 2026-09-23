# Duplicate Question Pair Detection

An NLP-based machine learning project for detecting whether two questions have the same meaning.

## Overview

Duplicate questions are questions that are worded differently but ask essentially the same thing.

This project explores different feature representations and machine learning approaches for classifying question pairs as:

- `1` → Duplicate
- `0` → Not Duplicate

The project was developed incrementally, starting with a Bag-of-Words baseline and then adding handcrafted text similarity features and preprocessing.

## Dataset

The project uses a Quora-style question pair dataset containing:

- `id`
- `qid1`
- `qid2`
- `question1`
- `question2`
- `is_duplicate`

For model experimentation, a sample of **30,000 question pairs** was used.

The dataset can be accessed here:

[Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)

## Project Workflow

### 1. Initial EDA

`initial_EDA.ipynb`

Performed exploratory data analysis including:

- Missing-value analysis
- Duplicate-row analysis
- Duplicate vs. non-duplicate distribution
- Repeated-question analysis
- Number of unique questions

The dataset contained **537,933 unique questions** and **111,780 repeated questions**.

### 2. Bag-of-Words Baseline

`only-bow.ipynb`

Built a baseline model using:

- CountVectorizer
- 3,000 vocabulary features per question
- Random Forest Classifier

The Bag-of-Words baseline achieved:

**74.20% accuracy**

### 3. Basic Feature Engineering

`bow-with-basic-features.ipynb`

Added handcrafted features based on the structure of the two questions:

- Question length
- Number of words
- Common words
- Total words
- Word-share ratio

The Random Forest model achieved:

**76.83% accuracy**

### 4. Text Preprocessing & Advanced Features

`bow-with-preprocessing-and-advanced-features.ipynb`

Added text preprocessing including:

- Lowercasing and whitespace normalization
- Special-character normalization
- Number normalization
- Contraction expansion
- HTML removal
- Punctuation removal

Additional similarity features were created using:

**Token Features**
- Common non-stopword ratios
- Common stopword ratios
- Common token ratios
- First-word equality
- Last-word equality

**Length & String Features**
- Absolute length difference
- Mean question length
- Longest common substring ratio

**Fuzzy Matching Features**
- Fuzzy ratio
- Partial ratio
- Token sort ratio
- Token set ratio

In total, **22 handcrafted features** were combined with **6,000 Bag-of-Words features** (3,000 for each question).

The final feature matrix contained **6,022 input features**.

## Model Performance

| Approach | Model | Accuracy |
|---|---|---:|
| Bag-of-Words | Random Forest | 74.20% |
| BOW + Basic Features | Random Forest | 76.83% |
| BOW + Advanced Features | Random Forest | 78.47% |
| BOW + Advanced Features | XGBoost | 79.27% |

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- BeautifulSoup
- FuzzyWuzzy
- Distance
- Matplotlib
- Seaborn
- Jupyter Notebook

## Repository Structure

```text
Duplicate-Question-Pair-Detection/
│
├── README.md
├── initial_EDA.ipynb
├── only-bow.ipynb
├── bow-with-basic-features.ipynb
└── bow-with-preprocessing-and-advanced-features.ipynb
