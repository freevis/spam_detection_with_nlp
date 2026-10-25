
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    # Convert the document into TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform([text])

    # Get words and phrases
    words = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Sort keywords by score in descending order
    ranked_keywords = sorted(
        zip(words, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_keywords[:top_n]


# Main program
print("=" * 50)
print(" AUTOMATIC KEYWORD EXTRACTION USING NLP")
print("=" * 50)

document = input("\nEnter your document text:\n\n")

if len(document.split()) < 3:
    print("\nPlease enter a longer document.")
else:
    keywords = extract_keywords(document)

    print("\nTop Keywords and Their Scores:")
    print("-" * 40)

    for keyword, score in keywords:
        print(f"{keyword:<25} {score:.4f}")

