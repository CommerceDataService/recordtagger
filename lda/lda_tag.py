#!/usr/bin/python
# lda_tag.py
#
#
# Title:        Topic extraction using Latent Dirichlet Allocation
# Author:       Rebecca Bilbro
# Version:      2.0
# Date:         last updated 3/7/16
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
import pymongo
import requests
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#####################################################################
# Global Variables
#####################################################################
conn=pymongo.MongoClient()
db = conn.earthwindfire
noaa_coll = db.noaa_coll

n_features = 200000
n_topics = 50
n_top_words = 30

domain_stops = ["department","commerce","doc","noaa","national", "data", \
                "centers", "united", "states", "administration"]
stopwords = text.ENGLISH_STOP_WORDS.union(domain_stops)


#####################################################################
# Helper Functions
#####################################################################
def saveClusters(model, feature_names, n_top_words):
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

def getWords(feature_names,cluster_id):
    """
    Just return the words for a given cluster, with given feature_names.
    """
    words = []
    for topic_idx, topic in enumerate(lda.components_):
        if topic_idx == cluster_id:
            words.append(" ".join([tf_feature_names[i].encode('utf-8') for i in topic.argsort()[:-31:-1]]))
    return words

def printClusters(model, feature_names, n_top_words):
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

def loadData(URL,collection):
    """
    Loads the JSON data from URL, connects to MongoDB & enters dicts into collection
    """
    if collection.count() > 20000:
        print("%d records already loaded" % collection.count())
    else:
        r = requests.get(URL)
        json_data = r.json()
        for dataset in json_data:
            data ={}
            data["title"] = dataset["title"]
            data["description"] = dataset["description"]
            data["keywords"] = dataset["keyword"]
            collection.insert_one(data)
        print("Successfully loaded %d records into MongoDB." % collection.count())

def wrangleData(collection):
    """
    Reads in MongoDB documents, extracts and joins the content from the relevant
    fields for each record (keyword, title, description) and returns a list.
    """
    data_samples = []
    for entry in collection.find():
        title = " ".join(filter(lambda x: x.isalpha(), entry[u'title'].split()))
        description = " ".join(filter(lambda x: x.isalpha(), entry[u'description'].split()))
        keywords = " ".join(filter(lambda x: x.isalpha(), entry[u'keywords']))
        data_samples.append(title+" "+description+" "+keywords)
    return data_samples


if __name__ == '__main__':
    with open('lda_clusters_v2.csv', 'wb') as f1:
        writer = csv.writer(f1)
        writer.writerow(["cluster","top words"])

        # Start the clock
        t0 = time()

        # Load the data into MongoDB
        print("Checking to see if you have the data...")
        loadData("https://data.noaa.gov/data.json",noaa_coll)

        noaa_samples = wrangleData(noaa_coll)
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
        for x in saveClusters(lda, tf_feature_names, n_top_words):
            writer.writerow([str(x[0]),str(x[1])])

        # # You can also pring out the clusters if you want to see them
        # printClusters(lda, tf_feature_names, n_top_words)

    # Now match up the records with the best fit clusters & corresponding keywords
    with open('records_to_ldaclusters_v2.csv', 'wb') as f2:
        writer = csv.writer(f2)
        writer.writerow(["record_index","record_text","five_best_clusters","suggested_keywords"])

        # Restart the clock
        t0 = time()
        print("Finding the best keywords for each record and writing up results...")

        results = lda.transform(tf)
        for i in range(len(results)):
            try:
                best_results = (-results[i]).argsort()[:5]
                keywords = []
                for x in np.nditer(best_results):
                    keywords.extend(getWords(tf_feature_names, x))
                flattened = " ".join(keywords)
                writer.writerow([i, noaa_samples[i], best_results, flattened])
            #TODO => need to figure out the Unicode Error
            except UnicodeEncodeError: pass

        print("done in %0.3fs." % (time() - t0))
