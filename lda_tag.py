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
import json
import requests
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#####################################################################
# Global Variables
#####################################################################
n_features = 20000
n_topics = 200
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
        data_samples.append(title+" "+description+' '.join(filter(lambda x: x.isalpha(), entry[u'keyword']))+" ")
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

    # Extract raw term counts to compute term frequency.
    print("Extracting term frequency features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
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

    # Print out the clusters
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_clusters(lda, tf_feature_names, n_top_words)
