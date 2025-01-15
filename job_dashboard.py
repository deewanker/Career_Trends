import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
import plotly.express as px

# Load the dataset
@st.cache_data(ttl=3600)
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['published_date'] = pd.to_datetime(data['published_date'], errors='coerce')
    data['title'] = data['title'].fillna('Unknown')
    data['link'] = data['link'].fillna('Not Available')
    data['hourly_low'] = data['hourly_low'].fillna(data['hourly_low'].median())
    data['hourly_high'] = data['hourly_high'].fillna(data['hourly_high'].median())
    data['budget'] = data['budget'].fillna(data['budget'].median())
    data['country'] = data['country'].fillna('Unknown')
    return data

file_path = "C:/Users/vipin/OneDrive/Documents/Desktop/Project 8/all_upwork_jobs_2024-02-07-2024-03-24.csv"
jobs = load_data(file_path)

# Dashboard Title
st.title("Job Market Analysis Dashboard")
st.sidebar.title("Navigation")

# Sidebar Navigation
options = [
    "Data Overview", "Top Keywords", "Emerging Categories",
    "Forecasting", "Salary Analysis", "Job Recommendations"
]
choice = st.sidebar.radio("Go to:", options)

# Data Overview
if choice == "Data Overview":
    st.header("Dataset Overview")
    st.write(jobs.head(10))
    st.write("Shape of Dataset:", jobs.shape)
    st.write("Missing Values Percentage:")
    st.write(jobs.isnull().mean() * 100)

    # Outlier Detection
    st.subheader("Outlier Detection")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=jobs[['hourly_low', 'hourly_high', 'budget']], ax=ax)
    ax.set_title("Outliers in Numeric Columns")
    st.pyplot(fig)

# Top Keywords in Job Titles
elif choice == "Top Keywords":
    st.header("Top Keywords in Job Titles")
    st.write("Analyzing the most frequent keywords in job titles.")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(jobs['title'])
    keywords = vectorizer.get_feature_names_out()

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(jobs['title']))
    st.subheader("Word Cloud")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Top Keywords
    st.subheader("Top Keywords")
    st.write(pd.DataFrame({"Keyword": keywords}).head(20))

# Emerging Categories
elif choice == "Emerging Categories":
    st.header("Emerging Job Categories")
    category_trends = jobs.groupby(['title', jobs['published_date'].dt.to_period('M')]).size().reset_index(name='post_count')
    category_trends['published_date'] = category_trends['published_date'].dt.to_timestamp()

    category_growth = category_trends.groupby('title')['post_count'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1) if len(x) > 1 else 0
    ).sort_values(ascending=False)

    # Visualize Top Emerging Categories
    st.subheader("Top Emerging Categories")
    emerging_categories = category_growth.head(10)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(emerging_categories.index, emerging_categories.values, color='skyblue')
    ax.set_xlabel('Growth Rate')
    ax.set_title('Top Emerging Job Categories')
    st.pyplot(fig)

# Forecasting High-Demand Roles
elif choice == "Forecasting":
    st.header("Forecasting High-Demand Job Roles")
    role_trends = jobs.groupby(['title', jobs['published_date'].dt.to_period('M')]).size().reset_index(name='post_count')
    role_trends['published_date'] = role_trends['published_date'].dt.to_timestamp()
    role_trends_pivot = role_trends.pivot(index='published_date', columns='title', values='post_count').fillna(0)

    # User selects a job role to forecast
    job_role = st.selectbox("Select a Job Role", role_trends_pivot.columns)
    role_series = role_trends_pivot[job_role]

    # ARIMA parameters
    st.sidebar.subheader("ARIMA Model Parameters")
    p = st.sidebar.slider("p", 0, 5, 2)
    d = st.sidebar.slider("d", 0, 2, 1)
    q = st.sidebar.slider("q", 0, 5, 2)

    # Train-test split and ARIMA model
    train_size = int(len(role_series) * 0.8)
    train, test = role_series[:train_size], role_series[train_size:]
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))

    # Forecast Visualization
    forecast_dates = pd.date_range(train.index[-1], periods=len(forecast), freq='M')
    st.write(f"RMSE for {job_role}: {rmse:.2f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(role_series.index, role_series, label='Historical Data')
    ax.plot(forecast_dates, forecast, label='Forecast', color='orange')
    ax.set_title(f'Forecast for Job Role: {job_role}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Post Count')
    ax.legend()
    st.pyplot(fig)

# Salary Analysis
elif choice == "Salary Analysis":
    st.header("Salary Analysis by Region")
    region_salary = jobs.groupby('country')['budget'].mean().reset_index()
    region_salary_sorted = region_salary.sort_values(by='budget', ascending=False)

    # Bar Chart
    st.subheader("Average Salary by Region")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(region_salary_sorted['country'], region_salary_sorted['budget'], color='skyblue')
    ax.set_xlabel('Country')
    ax.set_ylabel('Average Salary')
    ax.set_title('Salary Comparison by Country')
    ax.set_xticklabels(region_salary_sorted['country'], rotation=45)
    st.pyplot(fig)

    # Interactive Map
    st.subheader("Salary Distribution on World Map")
    fig = px.choropleth(region_salary, 
                        locations='country', 
                        locationmode='country names', 
                        color='budget', 
                        title='Average Salary by Region',
                        color_continuous_scale='Viridis')
    st.plotly_chart(fig)

# Job Recommendations
elif choice == "Job Recommendations":
    st.header("Personalized Job Recommendations")
    jobs['combined_features'] = jobs['title'] + " " + jobs['country'] + " " + jobs['link']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(jobs['combined_features'])

    # User preferences
    title_pref = st.text_input("Preferred Job Title", "Python")
    country_pref = st.text_input("Preferred Country", "Remote")
    budget_pref = st.number_input("Minimum Budget", value=60000)

    user_query = f"{title_pref} {country_pref} {budget_pref}"
    query_vec = tfidf.transform([user_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    recommendations = jobs.iloc[top_indices][['title', 'country', 'budget']]

    st.subheader("Recommended Jobs")
    st.write(recommendations)

    # Download Option
    csv = recommendations.to_csv(index=False)
    st.download_button(label="Download Recommendations", data=csv, file_name='job_recommendations.csv', mime='text/csv')
