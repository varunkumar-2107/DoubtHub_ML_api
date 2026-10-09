import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import json


def text_preprocessing(text):
    text = text.lower()
    exclude = '!"#$%&\'()*,-./:;<=>?@[\\]^_`{|}~'
    for char in exclude:
            text = text.replace(char,'')
    return text



def solution(new_querry="",domain=""):

    base_dir = os.path.dirname(__file__)

    if domain == 'Backend':
        file_path = os.path.join(base_dir, 'data' ,'backend_qaa.csv')
    elif domain == 'Cyber Security':
        file_path = os.path.join(base_dir, 'data' ,'cybersecurity_qaa.csv')
    elif domain == 'DSA':
        file_path = os.path.join(base_dir, 'data' ,'dsa_qaa.csv')
    elif domain == 'Frontend':
        file_path = os.path.join(base_dir, 'data' ,'frontend_qaa.csv')
    elif domain == 'Programming Language':
        file_path = os.path.join(base_dir, 'data' ,'programming_language_qaa.csv')
    elif domain == 'ML/AI':
        file_path = os.path.join(base_dir, 'data' ,'aiml_qaa.csv')
    else:
        return json.dumps({"error": "Invalid domain specified."})
        

    df = pd.read_csv(file_path)
    # df = pd.read_csv('training/Clustering_data.csv.csv')
    # df = pd.read_csv(data_path)              
    df['question'] = df['question'].apply(text_preprocessing)

    # preprocess the new query
    new_querry = text_preprocessing(new_querry)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['question']).toarray()
    new_query_vector = vectorizer.transform([new_querry]).toarray()

    # train KMeans model
    model = KMeans(n_clusters=6, random_state=42)
    model.fit(X)

    # Predict cluster for new query
    cluster_label = model.predict(new_query_vector)


    # print(model.labels_)          [0 4 4 ... 4 1 3]
    # print(cluster_label)          [4]
    # print(cluster_label[0])        4


    # Get points in same cluster
    cluster_points = X[model.labels_ == cluster_label[0]]
    cluster_queries = df['question'][model.labels_ == cluster_label[0]]

    # print(cluster_points)      # give the points in the same cluster
    # print(cluster_queries)     # give the queries in the same cluster


    # Compute cosine distances
    distances = pairwise_distances(new_query_vector, cluster_points, metric="cosine")[0]


    # Get shortest queries
    nearest_indices = np.argsort(distances)[0]
    nearest_queries = cluster_queries.iloc[nearest_indices]


    # Build JSON with index : query
    result = {}
    result["Question"] = nearest_queries
    result["Answer"] = df['answer'][df['question'] == nearest_queries].values[0]

    # print(result)
    return result
    # return json.dumps(result)


# solution(new_querry="What is Linear Regression",domain="ML/AI")