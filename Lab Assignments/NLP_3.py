import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample dataset
data = {
    'text': ["This is a sample sentence.", "Another example of text cleaning!", "Text preprocessing is crucial in NLP."],
    'label': ['positive', 'negative', 'neutral']
}
df = pd.DataFrame(data)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['lemmatized_text'] = df['cleaned_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Stop Word Removal
stop_words = set(stopwords.words('english'))
df['processed_text'] = df['lemmatized_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Label Encoding
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# TF-IDF Representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

# Save Outputs
df.to_csv("processed_texts.csv", index=False)
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("Processing complete. Outputs saved.")
