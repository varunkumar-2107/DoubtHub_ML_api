import pandas as pd
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

def retrain(new_querry=""):
    
    df = pd.read_csv("data\Clustering_data.csv")             
    df["mixed"] = df["query"]+" "+df['label']+" "+df['topic']

    # preprocess the new query
    new_querry = text_preprocessing(new_querry)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["mixed"]).toarray()
    new_query_vector = vectorizer.transform([new_querry]).toarray()

    # train KMeans model
    model = KMeans(n_clusters=6, random_state=42)
    model.fit(X)

    # Predict cluster for new query
    cluster_label = model.predict(new_query_vector)[0]

    # Get points in same cluster
    cluster_points = X[model.labels_ == cluster_label]
    cluster_queries = df["query"][model.labels_ == cluster_label]

    # Compute cosine distances
    distances = pairwise_distances(new_query_vector, cluster_points, metric="cosine")[0]

    # Get top 5 nearest queries
    nearest_indices = np.argsort(distances)[:5]
    nearest_queries = cluster_queries.iloc[nearest_indices]

    # Build JSON with index : query
    result = {}
    i = 1
    for q in nearest_queries:
        result[i] = q
        i += 1
    # print(result)
    return json.dumps({"similar_queries": result})


# retrain(new_querry="What is react")