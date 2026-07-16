# 📩 SMS Spam Classifier

A machine learning-powered spam detection system that leverages a **Voting Classifier** ensemble (combining Extra Trees, Naive Bayes, and Random Forest) to identify and filter spam SMS and email messages with high precision and recall.

**🔗 Live Demo:** [donna7-spam-classifier.streamlit.app](https://donna7-spam-classifier.streamlit.app)
---

## 📌 Overview

This project builds a text classification pipeline that detects spam SMS messages. Raw text is cleaned and normalized through an NLP preprocessing pipeline, converted into numerical features, and classified using an ensemble machine learning model trained and evaluated on the classic SMS Spam Collection dataset.

## ✨ Features

- ✍️ Paste any SMS/text message into the app
- ⚡ Instant **Spam / Not Spam** prediction
- 📊 Model selected after comparing 11+ classifiers on accuracy, precision, recall, and F1-score
- 🧠 Ensemble (Voting Classifier) model for improved, more stable predictions

## 🖼️ Screenshots

Screenshots of the app in action are available in the [`Results`](./Results) folder.

## 🧠 How It Works

### 1. Dataset

The project uses the **SMS Spam Collection** dataset (`spam.csv`, in the [`dataset`](./dataset) folder), containing:

| Column | Description |
|---|---|
| `v1` (→ `target`) | Label indicating message type: `ham` or `spam` |
| `v2` (→ `text`) | Raw SMS message text |

### 2. Data Cleaning

- Dropped unused/empty columns
- Renamed `v1` → `target`, `v2` → `text`
- Encoded `target` as binary (`0` = ham, `1` = spam) using `LabelEncoder`
- Removed duplicate messages

### 3. Exploratory Data Analysis (EDA)

- Checked class balance — the dataset is **highly imbalanced** (far more ham than spam)
- Engineered `num_characters`, `num_words`, and `num_sentences` features and compared their distributions between spam and ham messages
- Found spam messages tend to be **longer and more consistently sized** than ham messages
- Visualized word frequency with **word clouds** and bar charts of the most common words in spam vs. ham messages

### 4. Text Preprocessing

Each message is transformed through the following pipeline (`transform_text` function):

1. Lowercasing
2. Tokenization (`nltk.word_tokenize`)
3. Removing non-alphanumeric tokens
4. Removing English stopwords and punctuation
5. **Stemming** using the Porter Stemmer (e.g. `dancing` → `danc`)

### 5. Feature Extraction

Preprocessed text is converted into numerical features using **`CountVectorizer`** (`max_features=3000`). A `TfidfVectorizer` was also evaluated but `CountVectorizer` produced better results overall for this dataset.

### 6. Model Selection

Eleven classifiers were trained and compared on accuracy, precision, recall, and F1-score, including Naive Bayes variants (Gaussian, Multinomial, Bernoulli), Logistic Regression, SVM, Decision Tree, Random Forest, KNN, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost.

The top individual performers with `CountVectorizer` were:

- **Extra Trees Classifier (ETC)**
- **Multinomial Naive Bayes (MNB)**
- **Random Forest (RF)**

### 7. Ensembling

- A **soft-voting `VotingClassifier`** combining ETC, MNB, and RF was built and compared against a **`StackingClassifier`** (same base models, Logistic Regression meta-model) and the standalone **BernoulliNB** model.
- The **Voting Classifier** was selected as the final model, achieving the best overall balance:

| Metric | BernoulliNB | Voting Classifier |
|---|---|---|
| Accuracy | 0.9836 | **0.9845** |
| Precision | **0.9919** | 0.9841 |
| Recall | 0.8841 | **0.8986** |
| F1-Score | 0.9349 | **0.9394** |

### 8. Deployment Preparation

The final trained pipeline is serialized with `pickle`:

| File | Contents |
|---|---|
| `vectorizer.pkl` | Fitted `CountVectorizer` used to transform raw text into features |
| `model.pkl` | Trained Voting Classifier (ETC + MNB + RF, soft voting) |

The Streamlit app loads these two files at startup and applies the same `transform_text` preprocessing to any new message before predicting.

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** — data processing
- **nltk** — tokenization, stopwords, Porter Stemmer
- **scikit-learn** — `CountVectorizer`, classifiers, `VotingClassifier`
- **xgboost** — gradient boosting classifier (evaluated during model selection)
- **matplotlib / seaborn / wordcloud** — EDA visualizations
- **Streamlit** — web app frontend
- **pickle** — model/vectorizer serialization

## 📁 Project Structure

```

├── app.py                       # Streamlit frontend
├── sms_spam_classifier.ipynb    # Data cleaning, EDA, preprocessing & model-building notebook
├── model.pkl                    # Serialized trained Voting Classifier
├── vectorizer.pkl               # Serialized fitted CountVectorizer
├── dataset/                     # Archived SMS spam dataset (spam.csv)
├── Results/                     # Screenshots of app results
├── requirements.txt             # Python dependencies
├── nltk.txt                     # NLTK corpora required at deploy time (punkt, stopwords)
├── setup.sh                     # Streamlit config setup 
├── Procfile                     # Process file for deployment
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Donna152/SMS-spam-classification.git
   cd sms-spam-classifier
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required NLTK data (also listed in `nltk.txt`):
   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords
   ```

4. Make sure `model.pkl` and `vectorizer.pkl` are present in the project root. If not, run through `sms_spam_classifier.ipynb` to regenerate them from `dataset/spam.csv`.

5. Run the app locally:
   ```bash
   streamlit run app.py
   ```

6. Open the URL shown in your terminal (typically `http://localhost:8501`).

## 🌐 Deployment

This app is built to be deployed on **Streamlit Community Cloud**.

To deploy your own copy:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select the repository, branch, and `app.py` as the entry point.
4. Streamlit Cloud will install dependencies from `requirements.txt` and download NLTK data listed in `nltk.txt` automatically.

## 📓 Notebook

`sms_spam_classifier.ipynb` contains the full, step-by-step pipeline used to build this system:
- Dataset loading, cleaning, and deduplication
- Exploratory data analysis (class balance, message-length distributions, word clouds, most common words)
- Text preprocessing pipeline (lowercasing, tokenization, stopword/punctuation removal, stemming)
- Feature extraction with `CountVectorizer` / `TfidfVectorizer`
- Training and comparison of 11 classification algorithms
- Voting and Stacking ensemble experiments and final model selection
- Serialization of the final `model.pkl` and `vectorizer.pkl` used by the Streamlit app

## 🔮 Possible Improvements

- Address class imbalance with techniques like SMOTE or class weighting
- Experiment with word embeddings (Word2Vec, GloVe) or transformer-based models (e.g. DistilBERT) for richer text representation
- Add confidence scores alongside the spam/ham prediction
- Expand the dataset with more recent SMS/spam examples

## 📄 License

This project is intended for educational and portfolio purposes. The dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), publicly available for spam-detection research.
