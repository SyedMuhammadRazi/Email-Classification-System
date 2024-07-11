import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def remove_stop_words(text):
    with open("stopwords.txt", "r", errors="surrogateescape") as f:
        stopwords = f.read().splitlines()
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(cleaned_words)

def update_stopwords(new_words, stopwords_file):
    # Read existing stopwords
    with open(stopwords_file, 'r', errors="surrogateescape") as f:
        existing_stopwords = set(f.read().splitlines())

    # Add new unique words to existing stopwords
    existing_stopwords.update(new_words)

    # Write updated stopwords to file
    with open(stopwords_file, 'w', errors="surrogateescape") as f:
        f.write('\n'.join(existing_stopwords))

# Load data from CSV file
data = pd.read_csv("combined_data.csv")

# Remove stopwords from text data
data['text'] = data['text'].apply(remove_stop_words)

# Split data into features (X) and target (y)
X = data['text']
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Multinomial Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example usage of update_stopwords function with new emails
new_emails = [
    "This is a new email with some common words like the, is, and new.",
    "Another email with more common words to update the stopwords list."
]

# Tokenize words from new emails
new_words = [word.lower() for email in new_emails for word in nltk.word_tokenize(email)]

# Filter out existing stopwords
with open("stopwords.txt", "r", errors="surrogateescape") as f:
    existing_stopwords = set(f.read().splitlines())
new_words = [word for word in new_words if word not in existing_stopwords]

# Update stopwords file with new words
update_stopwords(new_words, "stopwords.txt")
