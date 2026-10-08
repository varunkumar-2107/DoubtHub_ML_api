from pathlib import Path
import numpy as np
import pandas as pd
from typing import Annotated
import json
import joblib
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from training.Clustering import retrain , text_preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import streamlit as st

model = joblib.load('model/domain_classifier.joblib')
vectoriser = joblib.load('model/domain_classifier_vectorizer.joblib')

QA_CLUSTERS = 20                   
MAX_DISTANCE = 1.1    


@st.cache_resource
def read_root():
    return {"message": "Welcome to the Query Classification.Find the answers of your doubts"}

def recommend_querry(User_input):
    result = retrain(new_querry=User_input)
    data = json.loads(result) if isinstance(result, str) else result
    return list(data.get("similar_queries", {}).values())


@st.cache_resource
def predict_querry(User_input):
    input_data = [text.lower() for text in User_input]
    input_vector = vectoriser.transform(input_data).toarray()
    prediction = model.predict(input_vector)[0]
    if prediction.item() == 0:
        predict = 'Backend'
    elif prediction.item() == 1:
        predict = 'Cyber Security'
    elif prediction.item() == 2:
        predict = 'DSA'
    elif prediction.item() == 3:
        predict = 'Frontend'
    elif prediction.item() == 4:
        predict = 'ML/AI'
    elif prediction.item() == 5:
        predict = 'Programming Language'
    else:
        predict = 'Other'
    
    return JSONResponse(
        status_code=200,
        content={"prediction": predict},
    ) 

def get_domain(User_input):
    input_data = [text.lower() for text in User_input]
    input_vector = vectoriser.transform(input_data).toarray()
    prediction = model.predict(input_vector)[0]
    if prediction.item() == 0:
        predict = 'Backend'
        QA_PATH = "data/backend_qa.csv"
    elif prediction.item() == 1:
        predict = 'Cyber Security'
        QA_PATH = "data/cybersecurity_qa.csv"
    elif prediction.item() == 2:
        predict = 'DSA'
        QA_PATH = "data/dsa_qa.csv"
    elif prediction.item() == 3:
        predict = 'Frontend'
        QA_PATH = "data/frontend_qa.csv"
    elif prediction.item() == 4:
        predict = 'ML/AI'
        QA_PATH = "data/aiml_qa.csv"
    elif prediction.item() == 5:
        predict = 'Programming Language'
        QA_PATH = "data/programming_language_qa.csv"
    else:
        predict = 'Other'
    
    return predict

def get_path(domain):
    if domain == 'Backend':
        QA_PATH = "data/backend_qa.csv"
    elif domain == 'Cyber Security':
        QA_PATH = "data/cybersecurity_qa.csv"
    elif domain == 'DSA':
        QA_PATH = "data/dsa_qa.csv"
    elif domain == 'Frontend':
        QA_PATH = "data/frontend_qa.csv"
    elif domain == 'ML/AI':
        QA_PATH = "data/aiml_qa.csv"
    elif domain == 'Programming Language':
        QA_PATH = "data/programming_language_qa.csv"
    else:
        domain = 'other'
        print('There is no Q&A data availabel')
    
    return QA_PATH

@st.cache_resource
def build_qa_index():
    QA_PATH = get_path(domain)
    if not Path(QA_PATH).exists():
        return None
    df = pd.read_csv(QA_PATH).dropna(subset=["question", "answer"]).reset_index(drop=True)
    clean = df["question"].apply(text_preprocessing)
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(clean)
    kmeans = KMeans(n_clusters=QA_CLUSTERS, random_state=42, n_init=10).fit(X)
    return df, tfidf, X, kmeans

def get_answer(User_input, index):
    df, tfidf, X, kmeans = index
    vec = tfidf.transform([text_preprocessing(User_input)])
    if vec.nnz == 0:                       # none of the words are known
        return None
    cluster = kmeans.predict(vec)[0]
    members = np.where(kmeans.labels_ == cluster)[0]
    dist = pairwise_distances(vec, X[members], metric="euclidean")[0]
    best = int(dist.argmin())
    if dist[best] > MAX_DISTANCE:
        return None
    row = df.loc[members[best]]
    return {"question": row["question"], "answer": row["answer"], "distance": float(dist[best])}


st.set_page_config(page_title="Smart Doubt Solver", page_icon="💡", layout="centered")
st.title("💡 Smart College Doubt Solver")
st.caption("Enter your doubt: get its domain, an answer (if available) and similar queries.")

try:
    model, vectoriser
except Exception as e:
    st.error(f"Could not load model files. Check model / vectorizer.\n\n{e}")
    st.stop()

User_input = st.text_area("Your query", height=120)

if st.button("Solve my doubt", type="primary"):
    User_input = " ".join(User_input.split())
    if not User_input:
        st.warning("Please enter a query first.")
        st.stop()

    domain = get_domain(User_input)
    st.subheader("Predicted domain")
    st.success(domain)

    qa_index = build_qa_index()

    st.subheader("Answer")
    if domain != domain or domain == 'other':
        st.info(f"No answers are available for **{domain}** yet. The dataset only covers {QA_DOMAIN}.")
    elif qa_index is None:
        st.warning(f"Q&A dataset not found at `{QA_PATH}`.")
    else:
        ans = get_answer(User_input, qa_index)
        if ans is None:
            st.info("No close match found in the Q&A dataset for this query.")
        else:
            st.write(ans["answer"])
            st.caption(f"Matched question: *{ans['question']}*")

    st.subheader("Similar queries")
    try:
        with st.spinner("Finding similar queries..."):
            similar = recommend_querry(User_input)
        if similar:
            for i, q in enumerate(similar, 1):
                st.write(f"{i}. {q.strip()}")
        else:
            st.info("No similar queries found.")
    except Exception as e:
        st.error(f"retrain() failed: {e}")