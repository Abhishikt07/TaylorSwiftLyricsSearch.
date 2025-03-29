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

# Manually mapped album cover paths
album_covers = {
    "1989 (Taylor's Version)": r"D:\VS code\Taylore_swift\album cover\1989 (Taylor's Version).png",
    "Beautiful Eyes": r"D:\VS code\Taylore_swift\album cover\Beautiful Eyes.png",
    "Evermore": r"D:\VS code\Taylore_swift\album cover\Evermore.png",
    "Fearless (Taylor's Version)": r"D:\VS code\Taylore_swift\album cover\Fearless (Taylor's Version).png",
    "folklore": r"D:\VS code\Taylore_swift\album cover\folklore.jpg",
    "Lover": r"D:\VS code\Taylore_swift\album cover\Lover.jpg",
    "Midnights": r"D:\VS code\Taylore_swift\album cover\Midnights.png",
    "Red (Taylor's Version)": r"D:\VS code\Taylore_swift\album cover\Red (Taylor's Version).jpg",
    "reputation": r"D:\VS code\Taylore_swift\album cover\reputation.png",
    "Speak Now (Taylor's Version)": r"D:\VS code\Taylore_swift\album cover\Speak Now (Taylor's Version).png",
    "Taylor Swift": r"D:\VS code\Taylore_swift\album cover\Taylor Swift.png",
    "The Tortured Poets Department": r"D:\VS code\Taylore_swift\album cover\The Tortured Poets Department.jpg",
    "The Hunger Games": r"D:\VS code\Taylore_swift\album cover\artworks-000058368652-o8bupy-t500x500.jpg",
    "How Long Do You Think It's Gonna Last": r"D:\VS code\Taylore_swift\album cover\How_Long_Do_You_Think_It's_Gonna_Last_by_Big_Red_Machine.png",
    "Cats": r"D:\VS code\Taylore_swift\album cover\images (1).jpg",
    "Where The Crawdads Sing": r"D:\VS code\Taylore_swift\album cover\Where_the_Crawdads_Sing_(film).jpg",
    "Christmas Tree Farm": r"D:\VS code\Taylore_swift\album cover\images (2).jpg",
    "Fifty Shades Darker": r"D:\VS code\Taylore_swift\album cover\MV5BMTQ5NTk0Njg2N15BMl5BanBnXkFtZTgwNzk5Nzk3MDI@._V1_.jpg",
    "Miss Americana": r"D:\VS code\Taylore_swift\album cover\AAAABRheY2bI3qqsOiRPUNuvGPgPEkXSFdWt1cH7v1VT4Icogjx4HAnPCHA0C8Of2YxTjH3CfJwX8Irrl1qzbP4JHh3jRjypMFQv9hOcs1gSha4yWOgveqxyw5U8LnGz1fmVDM1X (2).webp",
    "Love Drunk": r"D:\VS code\Taylore_swift\album cover\ab67616d0000b273597e14bb684692973a0843ba.jpg",
    "Women in Music Part III": r"D:\VS code\Taylore_swift\album cover\Haim_-_Women_in_Music_Pt._III.png",
    "Two Lanes of Freedom": r"D:\VS code\Taylore_swift\album cover\Twolanes.jpg",
    "The Hannah Montana Movie": r"D:\VS code\Taylore_swift\album cover\Hannah-montana-movie-poster.jpg",
    "The Taylor Swift Holiday Collection": r"D:\VS code\Taylore_swift\album cover\images.jpg"
}

def get_album_cover(album_name):
    album_image_path = album_covers.get(album_name)
    if album_image_path and os.path.exists(album_image_path):
        return album_image_path
    else:
        return r"C:\Users\abhis\Pictures\album cover\black_background.png"  # Default black background


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

def get_album_cover(album_name):
    album_image_path = album_covers.get(album_name)

    # Default black background if album cover is missing
    if not album_image_path or not os.path.exists(album_image_path):
        return r"D:\VS code\Taylore_swift\album cover\black_background.png"  # Black background image
    
    return album_image_path



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

            # Get album cover and dominant color
            album_cover_path = get_album_cover(song_details['album'])

            # Display album cover
            st.image(album_cover_path, caption=f"{song_details['album']}", width=250)

        with col2:
            # Display full lyrics
            st.markdown(f"## 🎶 {song_details['title']} ({song_details['album']})")
            st.text_area("Lyrics:", song_details['lyrics'], height=400)
    else:
        st.warning("No matching songs found!")

if 'album_cover_path' in locals():
    st.write(f"Image Path: {album_cover_path}")
else:
    st.write("Image Path: Not available (No song selected or query is empty)")
