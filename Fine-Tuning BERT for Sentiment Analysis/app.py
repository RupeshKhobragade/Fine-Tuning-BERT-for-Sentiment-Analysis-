

# Install required packages
# !pip install streamlit pyngrok torch transformers

#Import libraries
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pyngrok import ngrok
import os

# Load the sentiment analysis model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# Function for sentiment prediction
def predict_sentiment(review):
    inputs = tokenizer(
        review,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()

    # Map the model output (1 to 5) to proper sentiment labels
    if predictions == 1:
        return "Very Negative 😡"
    elif predictions == 2:
        return "Negative 😞"
    elif predictions == 3:
        return "Neutral 😐"
    elif predictions == 4:
        return "Positive 🙂"
    elif predictions == 5:
        return "Very Positive 😃"
    else:
        return "Unknown Sentiment 🤔"


# Streamlit UI
st.title("Sentiment Analysis using BERT")
st.write("Enter a review to predict its sentiment:")

# User Input
review = st.text_input("Review")

if st.button("Predict Sentiment"):
    if review:
        sentiment = predict_sentiment(review)
        st.write(f"*Predicted Sentiment:* {sentiment}")
    else:
        st.warning("⚠ Please enter a review.")

# Run the Streamlit app
os.system("streamlit run app.py --server.port 8080")