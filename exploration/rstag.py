# rstag.py
#
#
#
# Title:        Mapping records to LDA-generated clusters
# Author:       Rebecca Bilbro + Star Ying
# Date:         1/20/16
# Organization: Commerce Data Service, U.S. Department of Commerce

#####################################################################
# Imports
#####################################################################
from __future__ import print_function, division, unicode_literals  # Not necessary for Python 3
from __future__ import

import re
import csv
import math
import json
import requests
from time import time
from sklearn.cluster import KMeans
from textblob import TextBlob as tb

#####################################################################
# Global Variables
#####################################################################
file_name = "lda_clusters.txt"
data_URL = "https://data.noaa.gov/data.json"

#####################################################################
# Pull records from URL & load clusters from disk
#####################################################################
def get_records(URL):
    """
    Takes JSON input data, extracts and joins the content from the relevant
    fields for each record (keyword, title, description) and returns a list.
    """
    r = requests.get(URL)
    json_data = r.json()
    data_samples = []
    for entry in json_data:
        title = " ".join(filter(lambda x: x.isalpha(), entry[u'title'].split()))
        description = " ".join(filter(lambda x: x.isalpha(), entry[u'description'].split()))
        keywords = " ".join(filter(lambda x: x.isalpha(), entry[u'keyword']))
        data_samples.append(title+" "+description+" "+keywords)
    return data_samples

def get_clusters(fname):
    with open(file_name, 'rb') as f:
        for line in f:
            yield line

#######################################################################
# Base TF-IDF Implementation
#######################################################################
def tf(word, blob):
    """
    Computes term frequency (number of times a word appears in a doc blob)
    normalized by dividing by the total number of words in blob
    """
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    """
    Returns the number of documents containing the word
    """
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    """
    Computes inverse document frequency (how common a word is among all docs
    in bloblist). The more common a word is, the lower its IDF.
    Take ratio of total number of documents to the number of documents
    containing word, then take the log of that. Add 1 to the divisor to
    prevent division by zero.
    """
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    """
    computes the TF-IDF score (product of tf and idf)
    """
    return tf(word, blob) * idf(word, bloblist)

def blobPrint(bloblist,n=3):
    """
    Get n suggested tags for each record and print to screen (defaults to top 3)
    """
    for i, blob in enumerate(bloblist):
        print("Suggested tags for record {}".format(noaa_recordlist[i]))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:n]:
            print("\tTag: {}, TF-IDF Score: {}".format(word, round(score, 5)))

#######################################################################
# Execute scoring
#######################################################################
def scoreSave(bloblist,path,n=5):
    """
    Save n suggested tags to csv (defaults to top 5)
    """
    with open(path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["noaa_record","suggested_tags","tfidf_score"])
        for i, blob in enumerate(bloblist):
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            writer.writerow([noaa_recordlist[i]])
            for word, score in sorted_words[:n]:
                writer.writerow(["",word,round(score, 5)])

if __name__ == "__main__":
    records = get_records(data_URL)
    for record in records:

    # Note: vect_dict.csv is a dictionary created via rubyexplr.py and contains
    # the first 500 NOAA json records stored with 'identifier' as the key
    # and text from the 'title', 'keyword' and 'description' fields as the values.
    noaa_recordlist = []
    noaa_bloblist = []
    with open('vect_dict.csv', 'rb') as csvfile:
        rdr = csv.reader(csvfile)
        for row in rdr:
            document_name = row[0]
            document_text = ''.join(['"""',row[1],'"""'])
            noaa_recordlist.append(document_name)
            noaa_bloblist.append(tb(document_text))

    # blobPrint(noaa_bloblist)

    scoreSave(noaa_bloblist,"test_tags.csv")
