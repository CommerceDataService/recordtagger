#########################################################################################
# Topic extraction w/ Latent Dirichlet Allocation
#########################################################################################
"""
In LDA, time complexity is proportional to (n_samples * iterations).
"""

from __future__ import print_function
from time import time

import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_features = 2000
n_topics = 200
n_top_words = 30


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % (topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


print("Loading dataset...")
t0 = time()

# TODO ingest directly from https://data.noaa.gov/data.json
with open('noaa_data.json', 'rb') as f:
    noaa = json.load(f)
data_samples = []
for entry in noaa:
    title = ' '.join(filter(lambda x: x.isalpha(), entry[u'title'].split()))
    description = ' '.join(filter(lambda x: x.isalpha(), entry[u'description'].split()))
    data_samples.append(title+" "+description+' '.join(filter(lambda x: x.isalpha(), entry[u'keyword']))+"\n")

print("done in %0.3fs." % (time() - t0))

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", \
             "against", "all", "almost", "alone", "along", "already", "also","although",\
             "always","am", "among", "amongst", "amoungst", "amount",  "an", "and", "another", \
             "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  \
             "at", "back","be","became", "because","become","becomes", "becoming", "been", \
             "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", \
             "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", \
             "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", \
             "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", \
             "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", \
             "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", \
             "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", \
             "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", \
             "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", \
             "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", \
             "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", \
             "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", \
             "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", \
             "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", \
             "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", \
             "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", \
             "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", \
             "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", \
             "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", \
             "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", \
             "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", \
             "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", \
             "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", \
             "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", \
             "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", \
             "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", \
             "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",\
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", \
             "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "", \
             "yours", "yourself", "yourselves", "the", "0", "1", "2", "3", "4", "5", "6", "7", "8", \
             "9","10","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",\
             "2011","2012","2013","2014","2015","department","commerce","doc","noaa","national",\
             "data", "centers", "united", "states", "administration"]


# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words=stopwords)
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the LDA model
print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (len(data_samples), n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
