import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textdistance import jaro_winkler, levenshtein
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import logging

logging.getLogger().setLevel(logging.ERROR)

use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# bert_embed = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")
# tokenizer = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
MODEL = 'use'

# Load the dataset
with open('/Users/ajay/Desktop/crux/data/train/company_tickers.json', 'r') as f:
    raw_data = json.load(f)

data = pd.DataFrame(raw_data.values())

def preprocess(text):
    # get rid of special characters
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return ' '.join(text.split()).lower()

data['company_name_processed'] = [preprocess(name) for name in data['title']]

def levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    return 1 - (levenshtein(s1, s2) / max_len)

def jaro_winkler_similarity(s1, s2):
    return jaro_winkler(s1, s2)

def jaccard_similarity(s1, s2):
    s1_words = set(s1.split())
    s2_words = set(s2.split())
    intersection = len(s1_words.intersection(s2_words))
    union = len(s1_words.union(s2_words))
    return intersection / union if union != 0 else 0

def get_embedding(text, model='use'):
    if model == 'use':
        return use_embed([text]).numpy()
    elif model == 'bert':
        input_dict = tokenizer([text])
        outputs = bert_embed(input_dict)
        pooled_output = outputs["pooled_output"]
        return pooled_output.numpy()
    else:
        raise ValueError(f"Unknown model: {model}")


data['embedding'] = [get_embedding(name, model=MODEL) for name in data['company_name_processed']]
data_embedding = np.array(data['embedding'].tolist())
data_embedding = data_embedding.reshape(data_embedding.shape[0], -1)

def search(query, top_n=3, model='use', string_similarity='levenshtein'):
    # Preprocess the query
    query_processed = preprocess(query)

    # Get query embedding
    query_embedding = get_embedding(query_processed, model=model)

    # Calculate semantic similarity
    if model=='bert':
        semantic_similarity_scores = cosine_similarity(query_pooled_output.reshape(1, -1), database_pooled_output)
    else:
        semantic_similarity_scores = cosine_similarity(query_embedding, data_embedding).flatten()

    # Calculate string similarity
    if string_similarity == 'jaro_winkler':
        string_similarity_scores = data['company_name_processed'].apply(jaro_winkler_similarity, s2=query_processed).values
    elif string_similarity == 'levenshtein':
        string_similarity_scores = data['company_name_processed'].apply(levenshtein_similarity, s2=query_processed).values
    elif string_similarity == 'jaccard':
        string_similarity_scores = data['company_name_processed'].apply(jaccard_similarity, s2=query_processed).values
    else:
        raise ValueError(f"Unknown string similarity: {string_similarity}")

    # Combine similarity scores
    # TODO: add weights here, or train some model to learn these 2 weights
    combined_scores = semantic_similarity_scores + string_similarity_scores

    # Get top_n matches
    top_n_indices = combined_scores.argsort()[-top_n:][::-1]
    top_n_matches = data.iloc[top_n_indices][['cik_str', 'ticker', 'title']].copy()

    # Add similarity scores to the results
    top_n_matches['semantic'] = semantic_similarity_scores[top_n_indices]
    top_n_matches['string'] = string_similarity_scores[top_n_indices]
    top_n_matches['combined'] = combined_scores[top_n_indices]


    top_matches = list(data.iloc[top_n_indices]['company_name_processed'].copy())

    return top_n_matches,top_matches

def eval():
    with open('data/eval/eval_clean.csv','r') as f:
        correct = 0
        total = 0
        reader = csv.reader(f)
        for row in reader:
            y,x = row
            _,yhat = search(x, string_similarity='jaro_winkler',model=MODEL)
            if y in yhat: correct += 1
            total += 1
            if total%250 == 0:
                print('total',total)
                print('correct',correct)
                print('accuracy',(correct/total)*100)
            # print('input:',x)
            # print('predicted:',yhat)
            # print('actual:',y)
        return (correct/total)*100



parser = argparse.ArgumentParser(description='Entity Search System')
parser.add_argument('--search', type=str, help='The search query', required=True)
args = parser.parse_args()
query = args.search
# Call the search function with the query argument
results, matches = search(query, model='use', string_similarity='jaro_winkler')
print(results)