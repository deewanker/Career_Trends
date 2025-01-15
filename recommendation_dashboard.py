import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow

@st.cache_data(ttl=3600)  # Cache the data to improve loading speed, with TTL of 1 hour
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        
        # Ensure there are no missing values and convert numeric columns to strings before concatenation
        data['combined_features'] = (
            data['title'].fillna('') + " " +
            data['description'].fillna('') + " " +
            data['country'].fillna('') + " " +
            data['budget'].fillna(0).astype(str)  # Convert 'budget' to string before concatenation
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
            # Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow server URI

def train_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    return tfidf, tfidf_matrix

# Streamlit Interface
st.title("Personalized Job Recommendation System")
file_path = st.sidebar.text_input("Upload Job Dataset CSV:", "C:/Users/vipin/OneDrive/Documents/Desktop/Project 8/all_upwork_jobs_2024-02-07-2024-03-24.csv")
if file_path:
    jobs = load_data(file_path)

    # Train and log model
    with mlflow.start_run(run_name="TFIDF_Job_Model"):
        tfidf, tfidf_matrix = train_model(jobs)
        mlflow.log_param("vectorizer", "TFIDF")
        mlflow.log_metric("n_samples", len(jobs))

    # User Input
    st.sidebar.subheader("Input Preferences")
    skills = st.sidebar.text_input("Skills", "Python")
    location = st.sidebar.text_input("Location", "Remote")
    salary = st.sidebar.number_input("Desired Salary", 50000)

    user_query = f"{skills} {location} {salary}"
    user_vector = tfidf.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]

    # Recommendations
    st.header("Recommended Jobs")
    recommendations = jobs.iloc[top_indices][['title', 'country', 'budget', 'link']]
    st.write(recommendations)
