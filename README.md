
# 🎙️ Accent Classification using Wav2Vec2

A Deep Learning project that uses HuggingFace's Wav2Vec2 model to classify English audio accents as **Canadian** 🇨🇦 or **British (England)** 🏴. Built with PyTorch, Transformers, and Librosa, this project achieves high accuracy on voice-based accent prediction.

---

## ✅ Results

- **Accuracy**: 97.83%
- **Precision**: 100.00%
- **Recall**: 95.35%
- **F1 Score**: 97.62%

---

## 🚀 Features

- Fine-tuned Wav2Vec2-Large-960h model
- Custom classification head for binary accent classification
- Achieved **97.83% Accuracy**, **100% Precision**, **97.62% F1 Score**
- Audio pre-processing with Librosa & Torchaudio
- Exportable results in CSV + visualizations
- Easy-to-use Jupyter Notebook interface

---

## 🛠️ Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Librosa
- Scikit-learn
- Matplotlib / Seaborn
- Git / GitHub

---

## 🧰 Setup Instructions

> Follow these steps to set up the project on your local machine.

### 1️⃣ Prerequisites

- Python 3.9 or higher
- Git installed ([Download Git](https://git-scm.com/downloads))

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/creativepurus/Accent_Classification.git
cd Accent_Classification
```

### 3️⃣ Create & Activate a Virtual Environment

```bash
# Create venv (do this only once)
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `en_model.ipynb` and run the cells step-by-step.

---

## 📊 Outputs

- `model_predictions.csv` → Stores predictions with confidence
- Graphs folder → Contains plots like confusion matrix, class-wise distribution
- Trained model saved to `wav2vec2-accent-classifier/`

---

## 🗂️ Project Structure

```
Accent_Classification/
├── en_model.ipynb              # Main notebook
├── requirements.txt            # Dependency list
├── wav2vec2-accent-classifier/ # Saved model & processor
├── model_predictions.csv       # Prediction output
├── Graphs/                     # Evaluation plots
├── data/                       # Audio + tsv files
└── README.md
```

---

## ⚠️ Notes

- Don’t forget to activate your virtual environment before installing packages or running the notebook.
- Make sure audio files and `validated.tsv` are placed correctly in the `en/` folder.

---

## 🧪 Future Improvements

- Add support for more accents (e.g., Australian, Indian, American)
- Deploy as a full-fledged web app
- Integrate with Flask/FastAPI for production API

---

## 👨‍💻 Author

Made with ❤️ by [Anand Purushottam](https://github.com/creativepurus)

[LinkedIn](https://www.linkedin.com/in/creativepurus/) | [GitHub](https://github.com/creativepurus)

### Love My Work ? You can 👉🏻 [![BUY ME A COFFEE](https://img.shields.io/badge/Buy%20Me%20a%20Coffee%20☕-%23FFDD00.svg?&style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/creativepuru)

---

## 📄 License

This project is licensed under the GNU GENERAL PUBLIC LICENSE.