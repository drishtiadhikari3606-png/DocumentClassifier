import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t.lower()) for t in nltk.word_tokenize(doc) if t.isalpha()]

print("Loading ALL 20 datasets (This might take 15-30 seconds)...")
# We removed the 'categories' filter so it grabs everything!
data = fetch_20newsgroups(subset='all') 
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("Building Robust AI Pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        stop_words='english', 
        max_features=15000, # Increased vocabulary size for the extra topics
        ngram_range=(1, 2)
    )),
    ('clf', MultinomialNB(alpha=0.005))
])

print("Training the 20-domain model...")
pipeline.fit(X_train, y_train)

acc = accuracy_score(y_test, pipeline.predict(X_test)) * 100
print(f"\nFine-Tuned Model Accuracy: {acc:.2f}%")

# Package both the model AND the 20 names together
saved_package = {
    'model': pipeline,
    'target_names': data.target_names
}
joblib.dump(saved_package, 'document_classifier.pkl')
print("Success: 20-Domain model saved to 'document_classifier.pkl'")