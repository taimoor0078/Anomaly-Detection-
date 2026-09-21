# Revenue Generation Analysis on E-Commerce Website

## Machine Learning and Pattern Recognition

This project develops a machine-learning classification pipeline to predict whether an online shopping session will generate revenue based on customer browsing behaviour and session characteristics.

The project focuses on **binary revenue prediction**, **data preprocessing**, **class-imbalance handling**, comparison of multiple classification algorithms, ensemble learning, and model evaluation using accuracy, precision, recall, and F1-score.

---

## Project Objective

The main objective is to predict whether a customer visiting an e-commerce website will generate revenue/purchase or not.

The prediction is based on browsing and session-level characteristics such as:

- Number of administrative pages visited
- Time spent on administrative pages
- Number of informational pages visited
- Time spent on informational pages
- Number of product-related pages visited
- Time spent on product-related pages
- Bounce rate
- Exit rate
- Page value
- Proximity to a special shopping day
- Month
- Visitor type
- Browser
- Region
- Operating system
- Traffic type
- Weekend status

The target variable is **Revenue**, represented as a binary value indicating purchase/revenue generation or no purchase.

---

## Problem Type

This is a **binary classification problem**.

```text
Customer Session
       │
       ▼
Browsing & Session Features
       │
       ▼
Data Preprocessing
       │
       ├── Numerical Scaling
       ├── Categorical Encoding
       └── Train/Test Split
       │
       ▼
Class Imbalance Analysis
       │
       ▼
Balancing Strategy
       │
       ├── Oversampling
       ├── Undersampling
       └── SMOTE
       │
       ▼
Classification Models
       │
       ├── Logistic Regression
       ├── Gaussian Naive Bayes
       ├── Decision Tree
       └── Random Forest
       │
       ▼
5-Fold Cross Validation
       │
       ▼
Model Evaluation
       │
       ├── Accuracy
       ├── Precision
       ├── Recall
       └── F1-Score
```

---

# Dataset

The project uses an e-commerce website session dataset.

The dataset contains **12,330 records and 18 columns**, as shown in the preprocessing/output section of the report.

The target column is:

| Column | Type | Description |
|---|---|---|
| `Revenue` | Boolean/Binary | Whether the session resulted in a purchase/revenue |

The dataset is imbalanced. The report's frequency analysis shows approximately:

- **84.5% non-buying sessions**
- **15.5% buying sessions**

Because the positive class is substantially smaller, class balancing techniques were incorporated into the machine-learning pipeline.

---

# Features

## Numerical Features

The following numerical variables are used:

| Feature | Type | Description | Preprocessing |
|---|---|---|---|
| `Administrative` | Continuous | Number of administrative pages | Standard Scaling |
| `Administrative_Duration` | Discrete | Time spent on administrative pages | Standard Scaling |
| `Informational` | Continuous | Number of informational pages | Standard Scaling |
| `Informational_Duration` | Continuous | Time spent on informational pages | Standard Scaling |
| `ProductRelated` | Discrete | Number of product-related pages | Standard Scaling |
| `ProductRelated_Duration` | Continuous | Time spent on product pages | Standard Scaling |
| `BounceRates` | Continuous | Percentage of users leaving after one page | Standard Scaling |
| `ExitRates` | Continuous | Percentage of exits from a page | Standard Scaling |
| `PageValues` | Continuous | Average value of pages before purchase | Standard Scaling |
| `SpecialDay` | Continuous | Closeness to special shopping days | Standard Scaling |

---

## Categorical Features

### Nominal Variables

| Feature | Description | Preprocessing |
|---|---|---|
| `Month` | Month of the session | One-Hot Encoding |
| `VisitorType` | New or returning visitor | One-Hot Encoding |

### Ordinal Variables

| Feature | Description | Preprocessing |
|---|---|---|
| `Browser` | Browser used | No additional preprocessing |
| `Region` | User geographic region | No additional preprocessing |
| `OperatingSystems` | Operating system used | No additional preprocessing |
| `TrafficType` | Traffic source | No additional preprocessing |

### Boolean Variable

| Feature | Description | Preprocessing |
|---|---|---|
| `Weekend` | Whether the session occurred on a weekend | No additional preprocessing |

---

# Data Preprocessing

The preprocessing pipeline contains several stages.

## 1. Numerical Feature Scaling

Numerical features are standard-scaled before model training.

The original project also includes an additional experiment using **MinMax scaling** for a Logistic Regression model.

The MinMax experiment reports:

```text
Original Dataset Shape:  (12330, 18)

Train Shape:             (9864, 17)
Test Shape:              (2466, 17)

After Encoding:
Train Shape:             (9864, 26)
Test Shape:              (2466, 26)

After Scaling:
Train Shape:             (9864, 26)
Test Shape:              (2466, 26)
```

---

## 2. One-Hot Encoding

Nominal categorical variables are converted into machine-readable numerical representations.

The variables encoded are:

- `Month`
- `VisitorType`

---

## 3. Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

The project uses `stratify=y` during the split so that the class distribution remains consistent between training and testing sets.

This is particularly important because the target variable is imbalanced.

---

# Class Imbalance

The initial class distribution shows that non-purchase sessions substantially outnumber purchase sessions.

```text
Non-Revenue Sessions ≈ 84.5%
Revenue Sessions     ≈ 15.5%
```

To address this imbalance, multiple strategies were investigated:

- Oversampling
- Undersampling
- SMOTE

An important part of the methodology is that balancing is applied **only to the training data**, while the test data remains unchanged.

This prevents the balancing process from leaking information into the test set and allows model performance to be evaluated on the original test distribution.

---

# Machine Learning Models

Four classification algorithms were implemented:

### 1. Logistic Regression

A linear classification model used as one of the baseline classifiers.

### 2. Gaussian Naive Bayes

A probabilistic classification algorithm based on the Naive Bayes assumption.

### 3. Decision Tree

A tree-based classification algorithm that learns decision rules from the input features.

### 4. Random Forest

An ensemble learning method consisting of multiple decision trees.

The Random Forest model was evaluated across the different balancing strategies and evaluation settings.

---

# Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many observations predicted as revenue-generating sessions were actually positive.

### Recall

Measures how many actual revenue-generating sessions were successfully identified.

### F1-Score

The harmonic mean of precision and recall.

The project also uses **5-fold cross-validation** in the reported experiments.

---

# Experimental Results

## Oversampling with Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.851 | 0.514 | 0.743 | 0.608 |
| Gaussian NB | 0.575 | 0.251 | 0.874 | 0.389 |
| Decision Tree | 0.867 | 0.574 | 0.539 | 0.556 |
| Random Forest | **0.896** | **0.673** | 0.636 | **0.654** |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.851 | 0.514 | 0.743 | 0.608 |
| Gaussian NB | 0.575 | 0.251 | 0.874 | 0.389 |
| Decision Tree | 0.867 | 0.574 | 0.539 | 0.556 |
| Random Forest | **0.896** | **0.673** | 0.636 | **0.654** |

---

# Oversampling Without Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.852 | 0.516 | 0.754 | 0.613 |
| Gaussian NB | 0.578 | 0.248 | 0.851 | 0.384 |
| Decision Tree | 0.865 | 0.569 | 0.529 | 0.548 |
| Random Forest | **0.893** | **0.661** | 0.628 | **0.644** |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.856 | 0.526 | 0.741 | 0.615 |
| Gaussian NB | 0.513 | 0.226 | 0.880 | 0.359 |
| Decision Tree | 0.865 | 0.569 | 0.529 | 0.548 |
| Random Forest | **0.893** | **0.661** | 0.628 | **0.644** |

---

# Undersampling with Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.839 | 0.487 | 0.757 | 0.592 |
| Gaussian NB | 0.601 | 0.262 | 0.869 | 0.403 |
| Decision Tree | 0.790 | 0.407 | 0.775 | 0.533 |
| Random Forest | **0.846** | 0.502 | **0.832** | 0.627 |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.839 | 0.487 | 0.757 | 0.592 |
| Gaussian NB | 0.601 | 0.262 | 0.869 | 0.403 |
| Decision Tree | 0.790 | 0.407 | 0.775 | 0.533 |
| Random Forest | **0.846** | 0.502 | **0.832** | 0.627 |

---

# Undersampling Without Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.849 | 0.508 | 0.736 | 0.601 |
| Gaussian NB | 0.550 | 0.241 | 0.887 | 0.379 |
| Decision Tree | 0.789 | 0.405 | 0.775 | 0.532 |
| Random Forest | **0.858** | 0.525 | **0.851** | **0.649** |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.849 | 0.508 | 0.736 | 0.601 |
| Gaussian NB | 0.550 | 0.241 | 0.887 | 0.379 |
| Decision Tree | 0.789 | 0.405 | 0.775 | 0.532 |
| Random Forest | **0.858** | 0.525 | **0.851** | **0.649** |

---

# SMOTE with Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.841 | 0.491 | 0.738 | 0.590 |
| Gaussian NB | 0.488 | 0.213 | 0.856 | 0.341 |
| Decision Tree | 0.853 | 0.522 | 0.615 | 0.565 |
| Random Forest | **0.889** | **0.628** | 0.694 | **0.659** |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.841 | 0.491 | 0.738 | 0.599 |
| Gaussian NB | 0.488 | 0.213 | 0.856 | 0.341 |
| Decision Tree | 0.853 | 0.522 | 0.615 | 0.565 |
| Random Forest | **0.889** | **0.628** | 0.694 | **0.659** |

---

# SMOTE Without Random State

### With Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.842 | 0.495 | 0.736 | 0.592 |
| Gaussian NB | 0.488 | 0.212 | 0.856 | 0.341 |
| Decision Tree | 0.853 | 0.522 | 0.615 | 0.565 |
| Random Forest | **0.889** | 0.627 | 0.694 | **0.659** |

### Without Cross Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.842 | 0.495 | 0.736 | 0.592 |
| Gaussian NB | 0.472 | 0.209 | 0.861 | 0.335 |
| Decision Tree | 0.854 | 0.508 | 0.589 | 0.545 |
| Random Forest | **0.889** | **0.631** | 0.698 | **0.658** |

---

# Overall Model Findings

Across the reported experiments, Random Forest achieved the strongest overall performance among the tested classifiers.

The report records a maximum Random Forest accuracy of **0.896**, with:

- Precision: **0.673**
- Recall: **0.636**
- F1-score: **0.654**

This result was obtained with **oversampling using a random state**.

The experiments also show that different balancing methods affect the trade-off between accuracy, precision, recall, and F1-score.

For example, Random Forest with undersampling achieved a recall of up to **0.851**, while oversampling produced the highest reported accuracy of **0.896**.

SMOTE-based experiments produced Random Forest F1-scores around **0.658–0.659**.

---

# Logistic Regression MinMax Experiment

An additional experiment was conducted to train a single Logistic Regression model using **MinMax scaling**, following the discussion referenced in the report.

## Without Cross Validation

| Method | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Oversampling (Random State) | 0.8179 | 0.4477 | 0.7513 | 0.5611 |
| SMOTE (Random State) | 0.8301 | 0.4688 | 0.7277 | 0.5703 |
| Undersampling | 0.7758 | 0.3868 | 0.7644 | 0.5136 |

## With 5-Fold Cross Validation

| Method | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Oversampling (Random State) | 0.8208 | 0.4528 | 0.7539 | 0.5658 |
| Undersampling (Random State) | 0.7700 | 0.3800 | 0.7700 | 0.5100 |
| SMOTE (Random State) | 0.8301 | 0.4688 | 0.7277 | 0.5703 |

The report additionally states that an oversampling experiment without random state achieved an accuracy of **0.856** for Logistic Regression.

---

# Architecture

The project architecture presented in the report contains the following stages:

```text
                 ┌───────────────────────┐
                 │      Dataset          │
                 │  12,330 Sessions      │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Data Preprocessing    │
                 │                       │
                 │ • Encoding            │
                 │ • Scaling             │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Class Balancing       │
                 │                       │
                 │ • Oversampling        │
                 │ • Undersampling       │
                 │ • SMOTE               │
                 └───────────┬───────────┘
                             │
                             ▼
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      Baseline Classifiers           Ensemble Model
              │                             │
      ┌───────┼────────┐              ┌─────┴─────┐
      │       │        │              │  Random   │
      ▼       ▼        ▼              │  Forest   │
   Logistic  Gaussian  Decision       └─────┬─────┘
  Regression   NB       Tree                  │
      │       │        │                     │
      └───────┴────────┴─────────────┬───────┘
                                     ▼
                              5-Fold Cross
                                Validation
                                     │
                                     ▼
                              Model Evaluation
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
                  Accuracy       Precision        Recall
                                     │
                                     ▼
                                 F1-Score
```

---

# Technologies and Tools

The project report references the following tools and technologies:

- Python
- Python Software Foundation
- Scikit-learn
- Google Colab
- OpenAI
- Pandas
- Machine learning classification algorithms
- Sampling/class-balancing techniques
- Cross-validation

The implementation uses standard Python machine-learning tooling for preprocessing, model training, sampling, and evaluation.

---

# Reproducible Workflow

A simplified implementation workflow is:

```python
# 1. Load dataset
data = pd.read_csv("CA1.csv")

# 2. Separate features and target
X = data.drop("Revenue", axis=1)
y = data["Revenue"]

# 3. Preprocess numerical and categorical features
#    - Scaling
#    - One-hot encoding

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y
)

# 5. Apply class balancing ONLY to training data
#    - Oversampling
#    - Undersampling
#    - SMOTE

# 6. Train classifiers
#    - Logistic Regression
#    - Gaussian Naive Bayes
#    - Decision Tree
#    - Random Forest

# 7. Evaluate
#    - Accuracy
#    - Precision
#    - Recall
#    - F1-score

# 8. Apply 5-fold cross-validation
```

---

# Key Findings

### Class Imbalance

The target variable is significantly imbalanced, with approximately 84.5% non-revenue sessions and 15.5% revenue sessions. This motivated the use of multiple balancing approaches.

### Preprocessing

Numerical features were standard-scaled, while nominal categorical features were one-hot encoded. Browser, Region, OperatingSystems, and TrafficType were retained without additional preprocessing according to the project's methodology.

### Balancing

The project investigated:

- Oversampling
- Undersampling
- SMOTE

Balancing was performed on the training data while the test dataset remained unchanged.

### Model Comparison

The project evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest

### Ensemble Learning

Random Forest produced the highest reported accuracy of **0.896** under the oversampling-with-random-state experiment.

### Logistic Regression

The additional MinMax-scaling experiment showed different performance depending on the balancing strategy, with the report recording an accuracy of **0.856** for Logistic Regression under oversampling without random state.

---

# Limitations and Scope

The report focuses on offline machine-learning evaluation of historical e-commerce session data.

The project does not establish real-time production deployment or real-time customer intervention. Its primary scope is to demonstrate how preprocessing, class-imbalance handling, classification algorithms, ensemble learning, and cross-validation can be applied to online purchase/revenue prediction.

The reported metrics should therefore be interpreted within the dataset and experimental settings documented in the project.

---

# Conclusion

This project implements a complete machine-learning pipeline for predicting whether an online shopping session will generate revenue.

The workflow includes:

1. Dataset analysis
2. Class-distribution analysis
3. Numerical feature scaling
4. Categorical feature encoding
5. Stratified train-test splitting
6. Training-only class balancing
7. Multiple classification algorithms
8. Random Forest ensemble learning
9. 5-fold cross-validation
10. Performance evaluation

The experiments demonstrate that handling class imbalance and selecting an appropriate ensemble model can substantially affect the performance of e-commerce purchase prediction.

The reported experiments show that Random Forest achieved the highest reported accuracy of **0.896**, while different sampling strategies produced different precision, recall, and F1-score trade-offs.

---

## Project Information

**Project Title:** Revenue Generation Analysis on E-Commerce Website  
**Subject:** Machine Learning and Pattern Recognition  
**Assessment:** Continuous Assessment 1 (CA1)  
**Student:** Taimoor Ahmad  
**Student ID:** 20054191  
**Programme:** MSc Artificial Intelligence  
**Institution:** Dublin Business School  
**Academic Year:** 2026–2027

---

## References / Tools Mentioned in the Project

- Python Software Foundation — Python
- Scikit-learn Developers — scikit-learn
- Google Colab
- OpenAI
- Pandas
- Class-balancing techniques including oversampling, undersampling, and SMOTE
