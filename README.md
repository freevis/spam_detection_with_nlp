# SMS Spam Detection Using NLP

A command-line spam classifier for SMS messages. It trains a Naive Bayes model on the SMS Spam Collection dataset using TF-IDF features, and lets you check a message by typing it in or by uploading a screenshot (text is pulled out with OCR).

## Features

- Trains on the SMS Spam Collection dataset (ham vs. spam)
- TF-IDF text vectorization with English stop-word removal
- Multinomial Naive Bayes classifier
- Prints accuracy, classification report, and confusion matrix on startup
- Interactive menu:
  1. Type a message and classify it
  2. Quit
  3. Upload a screenshot, extract text via OCR, confirm, then classify

## How It Works

1. **Load data**: reads `SMSSpamCollection` (tab-separated: `label<TAB>message`).
2. **Prepare data**: maps `ham` to 0 and `spam` to 1, then does a stratified 80/20 train-test split (`random_state=42`).
3. **Vectorize**: `TfidfVectorizer` (lowercase, English stop words) turns messages into numeric features.
4. **Train**: `MultinomialNB` is fit on the training set.
5. **Evaluate**: accuracy, precision/recall/F1 per class, and confusion matrix on the test set.
6. **Predict**: new text goes through the same vectorizer and model.
7. **Screenshot input**: `pytesseract` extracts text from the image, you confirm it, then it's classified.

## Requirements

- Python 3.8+
- Python packages:
  - `pandas`
  - `scikit-learn`
  - `pytesseract`
  - `Pillow`
- `tkinter` (ships with most Python installs; on Linux install `python3-tk`)
- **Tesseract OCR engine** (separate from the Python package, required for option 3)

### Installing Tesseract

| OS | Command / Link |
|----|----------------|
| Windows | Download the installer from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH` |
| macOS | `brew install tesseract` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |

On Windows, if Tesseract isn't on `PATH`, add this near the top of the script:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas scikit-learn pytesseract Pillow
```

## Dataset

Download the **SMS Spam Collection** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Extract it and place the file named `SMSSpamCollection` (no extension) in the **same folder as the Python script**. Format:

```
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
```

## Usage

```bash
python spam_detector.py
```

(Replace `spam_detector.py` with your actual file name.)

On launch the model trains and prints its performance, then shows the menu:

```
==================================================
                 MAIN MENU
==================================================
1. Check a message
2. Quit
3. Upload a screenshot

Enter your choice (1, 2, or 3):
```

### Example

```
Enter your message: Congratulations! You've won a free iPhone. Click here to claim now.

----------------------------------------
PREDICTION RESULT
----------------------------------------
Result: SPAM MESSAGE
Warning: This message may be unwanted.
----------------------------------------
```

### Screenshot mode

Choose option 3, pick an image (`.png`, `.jpg`, `.jpeg`, `.webp`) in the file dialog, review the extracted text, and enter `y` to classify it.

## Project Structure

```
.
├── spam_detector.py      # Main script
├── SMSSpamCollection     # Dataset (download separately)
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: Dataset file 'SMSSpamCollection' not found` | Put the dataset in the same folder as the script and run from that folder |
| `TesseractNotFoundError` | Install the Tesseract engine and make sure it's on `PATH` (or set `tesseract_cmd`) |
| File dialog doesn't open | `tkinter` is missing or you're on a machine without a display (e.g. SSH, some servers) |
| Empty or garbage OCR output | Use a clearer, higher-resolution screenshot with plain text |
| `UnicodeDecodeError` while loading data | Change `encoding="utf-8"` to `encoding="latin-1"` in `pd.read_csv` |

## Limitations

- The dataset is small and old (English SMS, mid-2000s style), so modern spam (phishing links, UPI/OTP scams, non-English or Hinglish/Marathi text) may be misclassified.
- OCR errors directly hurt prediction quality.
- Bag-of-words ignores word order and context.
- The model is trained on every run (no saved model), so startup re-trains each time.

## Possible Improvements

- Save/load the model with `joblib` instead of retraining on each run
- Add n-grams, or try Logistic Regression / SVM and compare
- Show spam probability using `predict_proba`
- Build a web UI (Streamlit / Flask)
- Train on a larger, more recent, multilingual dataset

## Tech Stack

Python, pandas, scikit-learn, pytesseract, Pillow, tkinter

## License

Add your license here (e.g. MIT).
