# 🔎 TruthLens AI — Transformer-Based Fake News Detection

TruthLens AI is a **Transformer-based NLP application** that classifies news articles as **REAL** or **FAKE** using a fine-tuned **DistilBERT** model.

The project uses **Hugging Face Transformers, PyTorch, Scikit-learn, and Streamlit** to provide an end-to-end workflow covering data preprocessing, Transformer fine-tuning, evaluation, inference, and an interactive web interface.

---

## 🚀 Features

- Fine-tuned **DistilBERT** for binary text classification
- REAL / FAKE news prediction
- Prediction confidence score
- REAL and FAKE class probabilities
- Interactive **Streamlit dashboard**
- Transformer tokenization and attention masks
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Complete Jupyter Notebook workflow
- Local model saving and inference

---

## 🧠 Model Architecture

```text
News Article
      ↓
DistilBERT Tokenizer
      ↓
Input IDs + Attention Mask
      ↓
Fine-Tuned DistilBERT
      ↓
Classification Head
      ↓
Softmax Probabilities
      ↓
REAL / FAKE Prediction
      ↓
Streamlit Dashboard
```

---

## 📊 Model Performance

The current evaluation produced approximately:

| Metric | Score |
|---|---:|
| Accuracy | 97.47% |
| Precision | 0.97–0.98 |
| Recall | 0.97–0.98 |
| F1-Score | 0.97 |
| Evaluation Samples | 1,267 |

### Confusion Matrix

| Actual | Predicted REAL | Predicted FAKE |
|---|---:|---:|
| REAL | 624 | 10 |
| FAKE | 22 | 611 |

> **Note:** These results are from the current evaluation split. They should only be described as final test results if the samples came from a completely held-out test set.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Transformer:** DistilBERT
- **Deep Learning:** PyTorch
- **NLP:** Hugging Face Transformers
- **Dataset Processing:** Hugging Face Datasets
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Frontend:** Streamlit
- **Development:** Jupyter Notebook, VS Code
- **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
TruthLens_AI/
│
├── app.py
├── predict.py
├── train.py
├── TruthLens_AI.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── news.csv
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `train.py` | DistilBERT training and evaluation |
| `predict.py` | Loads the trained model and performs predictions |
| `TruthLens_AI.ipynb` | Complete notebook workflow |
| `requirements.txt` | Python dependencies |
| `data/news.csv` | Training dataset |
| `model/` | Location for trained model and tokenizer |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd transformer-fake-news-detection
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## 📦 Required Libraries

The main dependencies are:

```text
torch
torchvision
transformers
datasets
accelerate
streamlit
pandas
scikit-learn
numpy
```

---

## 📚 Dataset Format

The dataset should contain two columns:

```csv
text,label
"News article text here",0
"Another news article here",1
```

Labels:

```text
0 = REAL
1 = FAKE
```

If your dataset contains string labels such as `REAL` and `FAKE`, convert them before training.

```python
label_mapping = {
    "REAL": 0,
    "FAKE": 1
}

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map(label_mapping)
)
```

> The quality and diversity of the training dataset directly affect model performance.

---

## 🏋️ Train the Model

Run:

```powershell
python train.py
```

The script fine-tunes DistilBERT and saves the trained model and tokenizer inside:

```text
model/
```

Training a Transformer can take significant time on a CPU. A GPU environment such as Google Colab is recommended for training.

---

## ▶️ Run the Streamlit Application

Make sure the trained model is available inside the `model/` directory.

Then run:

```powershell
python -m streamlit run app.py
```

Open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

---

## 🔍 Example

### Input

```text
Scientists have discovered a secret planet made entirely of gold
only 500 kilometers from Earth, according to anonymous researchers.
```

### Example Output

```text
Prediction: FAKE
Confidence: 96.8%

REAL Probability: 3.2%
FAKE Probability: 96.8%
```

> The exact confidence score depends on the trained model.

---

## 📓 Jupyter Notebook

Open:

```text
TruthLens_AI.ipynb
```

The notebook demonstrates the complete machine-learning workflow:

1. Load and inspect the dataset
2. Perform basic EDA
3. Preprocess labels
4. Split training and validation data
5. Load the DistilBERT tokenizer
6. Tokenize news articles
7. Load DistilBERT for sequence classification
8. Fine-tune the Transformer
9. Evaluate model performance
10. Generate a classification report
11. Generate a confusion matrix
12. Test custom news articles
13. Save the trained model and tokenizer

---

## ⚠️ Important Limitation

TruthLens AI is a **text classification model, not an independent fact-checking system**.

The model learns statistical patterns from its training data. Therefore, a prediction of `REAL` does not prove that a statement is factually correct, and a prediction of `FAKE` does not independently prove that it is false.

A production-level system should additionally use:

- Reliable evidence retrieval
- Source credibility analysis
- Claim extraction
- Evidence-based verification
- Multiple trusted information sources

---

## 🔮 Future Improvements

- Compare DistilBERT with BERT and RoBERTa
- Add a completely held-out test dataset
- Add ROC-AUC evaluation
- Add batch CSV prediction
- Add Explainable AI (XAI)
- Add evidence retrieval
- Add source credibility scoring
- Add real-time fact-checking APIs
- Deploy the Streamlit application online

---

## 🎯 Project Highlights

This project demonstrates practical experience with:

**Transformers • NLP • DistilBERT • PyTorch • Hugging Face • Model Fine-Tuning • Text Classification • Model Evaluation • Streamlit • Git/GitHub**

---

## 📌 Disclaimer

This project is intended for **educational and research purposes**. Model predictions should not be treated as definitive factual judgments.