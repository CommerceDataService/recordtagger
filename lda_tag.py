# lda_tag.py
#
#
# Title:        Topic extraction using Latent Dirichlet Allocation
# Author:       Rebecca Bilbro
# Date:         1/9/16
# Organization: Commerce Data Service, U.S. Department of Commerce


"""
Notes: In LDA, time complexity is proportional to (n_samples * iterations).
"""

#####################################################################
# Imports
#####################################################################
from __future__ import print_function  # Not necessary for Python 3
from time import time

import re
import csv
import json
import requests
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#####################################################################
# Global Variables
#####################################################################
n_features = 200000
n_topics = 200
n_top_words = 30

domain_stops = ["department","commerce","doc","noaa","national", "data", \
                "centers", "united", "states", "administration"]
stopwords = text.ENGLISH_STOP_WORDS.union(domain_stops)


#####################################################################
# Helper Functions
#####################################################################
def save_clusters(model, feature_names, n_top_words):
    """
    Takes the model, the names of the features, and the
    requested number of top words for each cluster, and
    returns each cluster.
    """
    for topic_idx, topic in enumerate(model.components_):
        t = "Topic #%d:" % (topic_idx+1)
        topwords = []
        x = topic.argsort()
        y = x[:-n_top_words - 1:-1]
        for i in y:
            topwords.append(feature_names[i].encode('utf-8'))
        yield t,topwords

def get_words(feature_names,cluster_id):
    """
    Just return the words for a given cluster, with given feature_names.
    """
    words = []
    for topic_idx, topic in enumerate(lda.components_):
        if topic_idx == cluster_id:
            words.append(" ".join([tf_feature_names[i].encode('utf-8') for i in topic.argsort()[:-31:-1]]))
    return words

def print_clusters(model, feature_names, n_top_words):
    """
    Takes the model, the names of the features, and the
    requested number of top words for each cluster, and
    prints out each cluster.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % (topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def load_data(URL):
    """
    Loads the data from URL and returns data in JSON format.
    """
    r = requests.get(URL)
    data = r.json()
    return data

def wrangle_data(json_data):
    """
    Takes JSON input data, extracts and joins the content from the relevant
    fields for each record (keyword, title, description) and returns a list.
    """
    data_samples = []
    for entry in json_data:
        title = " ".join(filter(lambda x: x.isalpha(), entry[u'title'].split()))
        description = " ".join(filter(lambda x: x.isalpha(), entry[u'description'].split()))
        keywords = " ".join(filter(lambda x: x.isalpha(), entry[u'keyword']))
        data_samples.append(title+" "+description+" "+keywords)
    return data_samples


if __name__ == '__main__':
    with open('lda_clusters.csv', 'wb') as f1:
        writer = csv.writer(f1)
        writer.writerow(["cluster","top words"])

        # Start the clock
        t0 = time()

        # Load the data
        print("Loading dataset...")
        noaa = load_data("https://data.noaa.gov/data.json")
        noaa_samples = wrangle_data(noaa)
        print("done in %0.3fs." % (time() - t0))

        # Restart the clock
        t0 = time()

        # Extract raw term counts to compute term frequency.
        print("Extracting term frequency features for LDA...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2),
                                        stop_words=stopwords)
        tf = tf_vectorizer.fit_transform(noaa_samples)
        print("done in %0.3fs." % (time() - t0))

        # Restart the clock
        t0 = time()

        # Fit the LDA model
        print("Fitting LDA model with term frequency features, n_samples=%d and n_features=%d..."
            % (len(noaa_samples), n_features))
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5, learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        # Save the clusters
        tf_feature_names = tf_vectorizer.get_feature_names()
        for x in save_clusters(lda, tf_feature_names, n_top_words):
            writer.writerow([str(x[0]),str(x[1])])

        # # You can also pring out the clusters if you want to see them
        # print_clusters(lda, tf_feature_names, n_top_words)

    # Now match up the records with the best fit clusters & corresponding keywords
    with open('records_to_ldaclusters.csv', 'wb') as f2:
        writer = csv.writer(f2)
        writer.writerow(["record_index","record_text","five_best_clusters","suggested_keywords"])

        # Restart the clock
        t0 = time()

        results = lda.transform(tf)
        for i in range(len(results)):
            try:
                best_results = (-results[i]).argsort()[:5]
                keywords = []
                for x in np.nditer(best_results):
                    keywords.append(get_words(tf_feature_names, x))
                writer.writerow([i, noaa_samples[i], best_results, keywords])
            except UnicodeEncodeError: pass

        print("done in %0.3fs." % (time() - t0))
