# nmf_tag.py
#
#
# Title:        Topic extraction using Non-negative Matrix Factorization
# Author:       Rebecca Bilbro
# Date:         1/9/16
# Organization: Commerce Data Service, U.S. Department of Commerce


"""
Notes: In NMF, time complexity is polynomial.
"""
#####################################################################
# Imports
#####################################################################
from __future__ import print_function # Not necessary for Python 3
from time import time

import re
import json
import string
import requests
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

#####################################################################
# Global Variables
#####################################################################
n_features = 1000
n_topics = 20
n_top_words = 30

domain_stops = ["department","commerce","doc","noaa","national", "data", \
                "centers", "united", "states", "administration"]
stopwords = text.ENGLISH_STOP_WORDS.union(domain_stops)

#####################################################################
# Helper Functions
#####################################################################
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
        title = ' '.join(filter(lambda x: x.isalpha(), entry[u'title'].split()))
        description = ' '.join(filter(lambda x: x.isalpha(), entry[u'description'].split()))
        keywords = " ".join(filter(lambda x: x.isalpha(), entry[u'keyword']))
        data_samples.append(title+" "+description+" "+keywords)
    return data_samples

if __name__ == '__main__':
    # Start the clock
    t0 = time()

    # Load the data
    print("Loading dataset...")
    noaa = load_data("https://data.noaa.gov/data.json")
    noaa_samples = wrangle_data(noaa)
    print("done in %0.3fs." % (time() - t0))

    # Restart the clock
    t0 = time()

    # Extract term-frequency, inverse document-frequency features
    print("Extracting term-frequency, inverse document-frequency features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.95, min_df=2,
                                       stop_words=stopwords)

    tfidf = tfidf_vectorizer.fit_transform(noaa_samples)
    print("done in %0.3fs." % (time() - t0))

    # Restart the clock
    t0 = time()

    # Fit the non-negative matrix factorization model
    print("Fitting the NMF model with term-frequency, inverse document-frequency features."
          "n_samples=%d and n_features=%d..."
          % (len(noaa_samples), n_features))

    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    # Print out the clusters
    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_clusters(nmf, tfidf_feature_names, n_top_words)
