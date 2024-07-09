import streamlit as st
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the vectorizer
with open('tfidf.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the model
with open('best_model.pkl', 'rb') as model_file:
    voting_classifier = pickle.load(model_file)

# Stopwords and stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_input(input_text):
    input_text = input_text.lower()
    input_text = input_text.replace(r'[^\w\s]+', '')
    input_text = ' '.join([ps.stem(word) for word in input_text.split() if word not in stop_words])
    input_vectorized = vectorizer.transform([input_text]).toarray()
    return input_vectorized


def predict_class(input_text):
    processed_input = preprocess_input(input_text)
    prediction = voting_classifier.predict(processed_input)
    return prediction[0]


# Streamlit app
st.set_page_config(
    page_title="Suicide Detection App",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles
main_bg_color = "#E6E6FA"  # Lavender color
main_text_color = "#000000"  # Black text
button_bg_color = "#3498DB"  # Blue button
button_text_color = "#FFFFFF"  # White text on the button

# Colored background for the body
colored_style = f"""
    <style>
        .stApp {{
            background-color: {main_bg_color};
            color: {main_text_color};
        }}
    </style>
"""
st.markdown(colored_style, unsafe_allow_html=True)

# Title and header
st.title('Suicide Detection App')

# Input text box
user_input = st.text_area('Enter a text:')
if st.button('Analyze', key='analyze_button'):
    if user_input:
        st.write(f'**Input:** {user_input}')

        # Simulate a loading animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(1, 101):
            time.sleep(0.02)
            progress_bar.progress(i)
            status_text.text(f'Analyzing... {i}%')

        with st.spinner("Almost there..."):
            prediction = predict_class(user_input)
            time.sleep(1)  # Simulate some processing time

        # Display the result
        st.success(f'**Output:** {prediction}')

# Footer
st.markdown(
    """
    <style>
        footer {
            color: #000000;
            text-align: center;
            font-size: 12px;
            margin-top: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
