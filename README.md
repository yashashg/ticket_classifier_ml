# 🧠 Ticket Classification & Analysis - README

This repository includes a three-part NLP-based solution for processing and classifying IT support tickets. It covers issue type classification, named entity recognition (NER) integration, and urgency classification using machine learning models.

## 📁 Files Overview

### 1. **Issue_model.ipynb**
- **Purpose**: Classifies the type of issue in support tickets.
- **Model Used**: Logistic Regression
- **Workflow**:
  - Loads dataset `ai_dev_assignment_tickets_complex_1000.csv`.
  - Preprocesses text (e.g., lowercasing, stopword removal).
  - Vectorizes using TF-IDF.
  - Trains Logistic Regression model.
  - Evaluates model with accuracy and confusion matrix.
- **Pros**:
  - Simple and interpretable.
  - Fast to train and suitable for small/medium-sized datasets.
- **Cons**:
  - Doesn't handle semantics or context well.
  - Performance drops on imbalanced or highly varied text.


---

### 2. **Urgency_Binary_Classifier.ipynb**
- **Purpose**: Binary classifier to determine if a ticket is of **High** urgency or **Not High**.
- **Model Used**: Logistic Regression on embeddings
- **Feature Engineering**:
  - Uses `SentenceTransformer` (e.g., `'all-MiniLM-L6-v2'`) to convert text to embeddings.
  - Splits into training and test sets.
  - Evaluates with precision, recall, F1-score.
- **Pros**:
  - Embeddings capture sentence-level meaning.
  - Better at contextual urgency recognition than traditional TF-IDF.
- **Cons**:
  - Slightly heavier on computation.
  - Still limited to binary output (doesn't capture urgency gradation).

---

### 3. **NER_integration.ipynb**
- **Purpose**: Extracts named entities like dates, times, and customer-related details and loads the model to gice final output.
- **Model/Approach**:
  - Rule-based using regex and pattern matching.
  - Example use cases: Extracting dates, IDs, user names from ticket text.
- **Pros**:
  - Lightweight and doesn't require training.
  - Transparent and easily adaptable.
- **Cons**:
  - Brittle — regex fails if format changes.
  - Lacks understanding of contextual entity usage.

---

## ⚙️ Setup Instructions

### 1. **Clone the Repository**
```bash
git clone <repo_url>
cd <repo_directory>

pip install -r requirements.txt

2. Run Notebooks
You can open and run the notebooks using Jupyter:
