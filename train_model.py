import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import string

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_sentiment(s):
    s = str(s).strip().lower()
    if s in ['positive', 'negative', 'neutral', 'other']:
        return s
    elif s in ['happy', 'joy', 'excitement', 'admiration', 'thrill', 'contentment', 'love']:
        return 'positive'
    elif s in ['anger', 'sadness', 'fear', 'disgust', 'hate']:
        return 'negative'
    else:
        return 'neutral'

# Load and preprocess data
df = pd.read_csv('sentimentdataset.csv')
df = df[['Text', 'Sentiment', 'Platform', 'Retweets', 'Likes', 'Country']]
df.dropna(subset=['Text', 'Sentiment'], inplace=True)
df['Sentiment'] = df['Sentiment'].str.lower()
df['sentiment_label'] = df['Sentiment'].apply(map_sentiment)
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Train model
X = df['cleaned_text']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Save models
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print('\nModels saved successfully!')
print('- sentiment_model.pkl')
print('- vectorizer.pkl')