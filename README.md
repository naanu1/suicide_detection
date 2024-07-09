# Suicidal thoughts Detection System

This project is a Suicidal thoughts Detection System developed using machine learning models and deployed with Streamlit. The system analyzes text input to predict whether the content indicates suicidal intent.

## Features

- Text preprocessing including stemming and stopwords removal.
- Utilizes a voting classifier with GaussianNB, BernoulliNB, and MultinomialNB.
- Simulates loading animations and progress indicators for better user experience.
- Deployed as a web application using Streamlit.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/suicide-detection-system.git
    cd suicide-detection-system
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the NLTK stopwords and punkt datasets:
    ```sh
    python -m nltk.downloader stopwords punkt
    ```

4. Ensure you have the trained model and vectorizer files (`best_model.pkl` and `tfidf.pkl`) in the project directory.

## Running the Application

To run the Streamlit application, execute the following command:
```sh
streamlit run app.py
