# SMS Spam Detection Using NLP

A simple Python program that tells you whether an SMS message is **spam** or **not spam**. You can type a message, or upload a screenshot of one.

## What It Uses

- **TF-IDF** to convert text into numbers
- **Naive Bayes** to classify the message
- **Tesseract OCR** to read text from screenshots

---

## Installation (Step by Step)

### Step 1: Install Python

Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/).
Check it is installed:

```bash
python --version
```

### Step 2: Get the project

```bash
git clone https://github.com/freevis/spam_detection_with_nlp.git
cd https://github.com/freevis/spam_detection_with_nlp.git
```

Or download the ZIP from GitHub and extract it.

### Step 3: Install Python libraries

```bash
pip install pandas scikit-learn pytesseract Pillow
```

`tkinter` comes with Python on Windows and macOS. On Ubuntu/Debian, install it with:

```bash
sudo apt install python3-tk
```

### Step 4: Install Tesseract OCR (needed for screenshots)

This is a separate program, not a pip package.

- **Windows:** download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install it.
- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`

Check it works:

```bash
tesseract --version
```

**Windows only:** if the command is not found, add this line near the top of the Python file (after the imports):

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Step 5: Download the dataset

1. Download the **SMS Spam Collection** from [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
2. Extract the ZIP.
3. Copy the file named `SMSSpamCollection` (no extension) into the same folder as the Python file.

Your folder should look like this:

```
your-project/
├── spam_detector.py
├── SMSSpamCollection
└── README.md
```

---

## How to Run

```bash
python spam_detector.py
```

The program trains the model, shows its accuracy, and then opens a menu:

```
1. Check a message
2. Quit
3. Upload a screenshot
```

- **Option 1:** type a message and get the result.
- **Option 3:** pick an image, check the extracted text, and press `y` to classify it.

## Example Output

```
Result: SPAM MESSAGE
Warning: This message may be unwanted.
```

---

## Common Problems

| Problem | Fix |
|---------|-----|
| `Dataset file 'SMSSpamCollection' not found` | Keep the dataset in the same folder as the script and run from that folder |
| `TesseractNotFoundError` | Install Tesseract (Step 4) and add it to PATH |
| `ModuleNotFoundError` | Run the pip install command from Step 3 again |
| File picker doesn't open | `tkinter` is missing, or you have no display (SSH / server) |
| `UnicodeDecodeError` | In the script, change `encoding="utf-8"` to `encoding="latin-1"` |

## Limitations

- The dataset is old and English-only, so new scams or Hindi/Marathi messages may be misclassified.
- Blurry screenshots give bad OCR text, which gives bad results.

## Tech Stack

Python, pandas, scikit-learn, pytesseract, Pillow, tkinter
