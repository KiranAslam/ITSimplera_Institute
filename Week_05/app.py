import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from src.nlp_utils import predict_sentiment

st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="📦", layout="wide")

pages = ["Home", "Data Overview", "Sentiment Predictor"]
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", pages)

if selection == "Home":
    st.title("Amazon Review Sentiment Analysis")
    st.write(
        "This project uses NLP to analyze Amazon customer reviews and predict whether a review is positive or negative."
    )
    st.write("The dataset contains customer feedback text and sentiment labels, which are used to train and evaluate text classifiers.")
    st.info("The app uses a trained TF-IDF model and a cleaned text pipeline for prediction.")

elif selection == "Data Overview":
    st.title("Data Overview")
    st.write("Review sentiment distribution and word clouds from the training notebook are shown below.")
    from src.nlp_utils import POS_WORDCLOUD_PATH, NEG_WORDCLOUD_PATH, load_dataset, DATA_PATH
    import pandas as pd
    import matplotlib.pyplot as plt

    df = load_dataset(DATA_PATH)
    counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax, color=['#4C78A8', '#F58518'])
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sentiment')
    plt.xticks(rotation=0)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(POS_WORDCLOUD_PATH), caption='Positive Reviews')
    with col2:
        st.image(str(NEG_WORDCLOUD_PATH), caption='Negative Reviews')

else:
    st.title("Sentiment Predictor")
    st.write("Type a customer review and the app will predict whether it is positive or negative.")

    review_text = st.text_area("Enter a review", placeholder="Example: I love this product because it works really well")
    if st.button("Predict Sentiment"):
        if review_text.strip():
            sentiment, confidence = predict_sentiment(review_text)
            st.success(f"Predicted sentiment: {sentiment}")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        else:
            st.warning("Please enter a review before predicting.")
