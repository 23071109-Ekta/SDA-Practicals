
# Import required libraries
import pandas as pd
import numpy as np

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# 1. Sample Dataset (You can replace with your own dataset)
data = {
    'text': [
        "I love machine learning",
        "NLP is interesting",
        "I hate spam messages",
        "This is a great course",
        "I dislike bad content",
        "Amazing experience with AI",
        "Terrible service",
        "I enjoy learning new things",
        "Worst product ever",
        "Best experience",
        "Very bad quality",
        "Excellent work",
        "Not good at all",
        "I am very happy",
        "I am very disappointed",
        "Superb performance"
    ],
    'label': [1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)

# 2. Text Preprocessing Function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    return " ".join(filtered_tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# 3. Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nConfusion Matrix (Formatted):")
print("TN FP")
print("FN TP")
print(cm)

# 7. Test with New Sentence
new_text = ["I love this amazing course"]
new_text_clean = [preprocess_text(text) for text in new_text]
new_text_tfidf = vectorizer.transform(new_text_clean)

prediction = model.predict(new_text_tfidf)

print("\nNew Text Prediction:", "Positive" if prediction[0] == 1 else "Negative")