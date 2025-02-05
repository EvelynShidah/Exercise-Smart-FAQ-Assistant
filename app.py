import streamlit as st
import pandas as pd
import numpy as np
#import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    df["Question_Embedding"] = df["Question_Embedding"].apply(lambda x: np.array(eval(x)))
    return df

df = load_data()

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Streamlit UI
st.title("Heart, Lung, and Blood Health Q&A")
st.write("Ask any health-related question, and I'll try my best to find an answer!")

user_question = st.text_input("Enter your question:")
if st.button("Get Answer") and user_question:
    user_embedding = model.encode(user_question).reshape(1, -1)
    embeddings_matrix = np.vstack(df["Question_Embedding"].values)
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, embeddings_matrix)[0]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    # Define threshold for a relevant answer
    threshold = 0.75
    if best_score > threshold:
        st.subheader("Answer:")
        st.write(df.loc[best_match_idx, "Answer"])
        st.write(f"Similarity Score: {best_score:.2f}")
    else:
        st.write("I apologize, but I don't have information on that topic yet. Could you please ask another question?")

# Clear button
if st.button("Clear"):
    st.experimental_rerun()

# FAQs section
st.sidebar.header("Common Questions")
for question in df["Question"].sample(5):
    if st.sidebar.button(question):
        user_question = question
        st.experimental_rerun()

# User feedback
st.subheader("Was this answer helpful?")
feedback = st.radio("", ["Yes", "No"], index=None, horizontal=True)
if feedback == "Yes":
    st.success("Thanks for your feedback!")
elif feedback == "No":
    st.warning("We're always improving. Thanks for your input!")
