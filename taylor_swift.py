import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv(r"C:\Users\abhis\Downloads\songs_cleaned.csv")

print(df.head())
print(df.info())

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        return " ".join([word.lower() for word in text.split() if word.lower() not in stop_words])
    return ""

df['clean_lyrics_new'] = df['Cleaned Lyrics'].apply(clean_text)
df['clean_title'] = df['Title'].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_lyrics_new'] + " " + df['clean_title'])


def search_song(query, top_n=5):
    query_cleaned = clean_text(query)  
    query_vector = vectorizer.transform([query_cleaned])  

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][['Title', 'Album', 'Lyrics']]
    return results

query = "lover"

df = pd.read_csv(r"C:\Users\abhis\Downloads\songs_cleaned.csv")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        return " ".join([word.lower() for word in text.split() if word.lower() not in stop_words])
    return ""

df['clean_lyrics_new'] = df['Cleaned Lyrics'].apply(clean_text)
df['clean_title'] = df['Title'].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_lyrics_new'] + " " + df['clean_title'])

with open("tfidf_model.pkl", "wb") as model_file:
    pickle.dump(tfidf_matrix, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

df.to_csv("cleaned_songs.csv", index=False)

print("✅ TF-IDF model and vectorizer saved successfully!")


