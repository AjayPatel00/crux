import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textdistance import jaro_winkler, levenshtein
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import csv


use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
bert_embed = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")
tokenizer = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

with open('data/train/company_tickers.json', 'r') as f:
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

model = 'use'

data['embedding'] = [get_embedding(name, model=model) for name in data['company_name_processed']]
data_embedding = np.array(data['embedding'].tolist())
data_embedding = data_embedding.reshape(data_embedding.shape[0], -1)

def search_basic(query, data, top_n=3, string_similarity='levenshtein'):
    # Preprocess the query
    query_processed = preprocess(query)

    # Calculate string similarity
    if string_similarity == 'jaro_winkler':
        string_similarity_scores = data['company_name_processed'].apply(jaro_winkler_similarity, s2=query_processed).values
    elif string_similarity == 'levenshtein':
        string_similarity_scores = data['company_name_processed'].apply(levenshtein_similarity, s2=query_processed).values
    elif string_similarity == 'jaccard':
        string_similarity_scores = data['company_name_processed'].apply(jaccard_similarity, s2=query_processed).values
    else:
        raise ValueError(f"Unknown string similarity: {string_similarity}")

    string_similarity_scores

    # Get top_n matches
    top_n_indices = string_similarity_scores.argsort()[-top_n:][::-1]
    top_n_matches = data.iloc[top_n_indices][['cik_str', 'ticker', 'title']].copy()

    top_matches = list(data.iloc[top_n_indices]['company_name_processed'].copy())

    # Add similarity scores to the results
    top_n_matches['string'] = string_similarity_scores[top_n_indices]

    return top_n_matches,top_matches

def eval():
    with open('data/eval/eval_clean.csv','r') as f:
        correct = 0
        total = 0
        reader = csv.reader(f)
        for row in reader:
            y,x = row
            _,yhat = search_basic(x, data, string_similarity='levenshtein')
            if y in yhat: correct += 1
            total += 1
            if total%500 == 0:
                print('total',total)
                print('correct',correct)
                print('accuracy',(correct/total)*100)
            # print('input:',x)
            # print('predicted:',yhat)
            # print('actual:',y)
        return (correct/total)*100


eval()
print("lev")


# accuracy for jaro_winkler: accuracy 72.15384615384616 at 18200 samples processed
# accuracy for levenshtein: accuracy 60.099999999999994 at 18000 samples processed
