
import pandas as pd
import pytesseract

from PIL import Image
from tkinter import Tk, filedialog

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

print("=" * 50)
print("       SMS SPAM DETECTION USING NLP")
print("=" * 50)

try:
    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
except FileNotFoundError:
    print("\nERROR: Dataset file 'SMSSpamCollection' not found.")
    print("Place it in the same folder as this Python file.")
    raise SystemExit(1)

print("\nDataset loaded successfully!")
print("Total messages:", len(data))


# ==================================================
# 2. PREPARE DATA
# ==================================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==================================================
# 3. TF-IDF TEXT VECTORIZATION
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==================================================
# 4. TRAIN MODEL
# ==================================================

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("\nModel training completed successfully!")


# ==================================================
# 5. MODEL PERFORMANCE
# ==================================================

predictions = model.predict(X_test_tfidf)

print("\n" + "*" * 50)
print("             MODEL PERFORMANCE")
print("*" * 50)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ==================================================
# 6. PREDICTION FUNCTION
# ==================================================

def check_message(message):
    """Classify text using the trained TF-IDF vectorizer and model."""

    if not message.strip():
        print("No readable text was provided.")
        return

    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

    print("\n" + "-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)

    if prediction == 1:
        print("Result: SPAM MESSAGE")
        print("Warning: This message may be unwanted.")
    else:
        print("Result: NOT SPAM MESSAGE")
        print("The model classified this as legitimate.")

    print("-" * 40)


# ==================================================
# 7. MAIN MENU
# ==================================================

while True:
    print("\n" + "-" * 50)
    print("                 MAIN MENU")
    print("-" * 50)
    print("1. Check a message")
    print("2. Quit")
    print("3. Upload a screenshot")

    choice = input("\nEnter your choice (1, 2, or 3): ").strip()

    # OPTION 1: ENTER MESSAGE
    if choice == "1":
        message = input("\nEnter your message: ").strip()
        check_message(message)

    # OPTION 2: QUIT
    elif choice == "2":
        print("\nThank you for using the Spam Detection System!")
        print("Exiting program. Goodbye!")
        break

    # OPTION 3: UPLOAD SCREENSHOT
    elif choice == "3":
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        image_path = filedialog.askopenfilename(
            title="Select a screenshot",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

        if not image_path:
            print("No screenshot selected. Returning to menu.")
            continue

        try:
            # Extract text from the screenshot using OCR
            with Image.open(image_path) as image:
                message = pytesseract.image_to_string(image).strip()

            if not message:
                print("No readable text found in the screenshot.")
                continue

            print("\nExtracted text:")
            print("-" * 40)
            print(message)
            print("-" * 40)

            confirm = input(
                "\nDo you want to spam detect? (y/n): "
            ).strip().lower()

            if confirm == "y":
                check_message(message)
            else:
                print("Classification cancelled.")

        except Exception as error:
            print("Could not process screenshot:", error)

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
