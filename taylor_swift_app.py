import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# -------------------- Authentication System -------------------- #

# Define correct credentials
USERNAME = "rohit saraf"
PASSWORD = "gayatri"

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Function to authenticate user
def login():
    st.title("🔒 Login to Access Taylor Swift Song Search 🎀")
    
    username = st.text_input("👤 Username")
    show_hint_user = st.button("Hint for username")
    if show_hint_user:
        st.info("Hint:The best guy you ever met, and the one you are absolutely in love with")

    password = st.text_input("🔑 Password", type="password")
    show_hint_pass = st.button("Hint for password")
    if show_hint_pass:
        st.info("Hint: a sacred chant, sung with grace, Even Shri Krishna speaks of it's trace!")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.success("✅ Login successful! Redirecting...")
            st.session_state.logged_in = True
            st.session_state.login_time = time.time()  # Store login timestamp
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid username or password!")

# Function to check session expiry (30 days)
def is_session_expired():
    if "login_time" in st.session_state:
        elapsed_time = time.time() - st.session_state.login_time
        return elapsed_time > (30 * 24 * 60 * 60)  # 30 days in seconds
    return True  # If no login time, session is expired

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.pop("login_time", None)
    st.rerun()


# Check session validity
if not st.session_state.logged_in or is_session_expired():
    login()
    st.stop()


# Get the absolute path to the CSV file within the project folder
csv_file = os.path.join(os.path.dirname(__file__), "songs_cleaned.csv")

# Load the dataset
df = pd.read_csv(csv_file)

# Load saved TF-IDF model and vectorizer
with open("tfidf_model.pkl", "rb") as model_file:
    tfidf_matrix = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Function to search songs
def search_song(query, top_n=5):
    query_vector = vectorizer.transform([query])  
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['Title', 'Album', 'Lyrics']].copy()
    results['index'] = top_indices  
    return results

# Function to fetch full lyrics
def get_full_lyrics(song_index):
    song = df.iloc[song_index]
    return {"title": song['Title'], "album": song['Album'], "lyrics": song['Lyrics']}


# ----------------------------------- Streamlit UI ---------------------------------------------

# Set page title
st.set_page_config(page_title="Taylor Swift Song Search 💅🏻🎀", layout="wide")

# Header
st.markdown("### To: Gayatri Ji 🛐❤️", unsafe_allow_html=True)
st.title("Taylor Swift Song Search 🎀")

# Small logout button at the top-right
st.sidebar.button("🚪 Logout", on_click=logout)

# Search box
query = st.text_input("Search by lyrics or song title:")

if query:
    results = search_song(query)

    if not results.empty:
        col1, col2 = st.columns([1, 3])

        with col1:
            selected_title = st.selectbox("Select a song:", results['Title'].tolist())

            # Find the index of selected song
            selected_index = results[results['Title'] == selected_title]['index'].values[0]
            song_details = get_full_lyrics(selected_index)

        with col2:
            # Display full lyrics
            st.markdown(f"## 🎶 {song_details['title']} ({song_details['album']})")
            st.text_area("Lyrics:", song_details['lyrics'], height=400)
    else:
        st.warning("No matching songs found!")

