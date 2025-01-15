import streamlit as st

st.title("Personalized Job Recommendation Engine")

skills = st.text_input("Enter your skills (comma-separated):")
location = st.text_input("Preferred location:")
desired_salary = st.number_input("Desired minimum salary:", min_value=0)

if st.button("Recommend Jobs"):
    user_preferences = {"skills": skills, "location": location, "desired_salary": str(desired_salary)}
    recommendations = recommend_jobs(user_preferences) # type: ignore

    st.write("Recommended Jobs:")
    st.write(recommendations)