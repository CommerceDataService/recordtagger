# rubytag.py

##########################################################################
# Imports
##########################################################################
import csv
import json
import string
import collections
from pprint import pprint

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

##########################################################################
# Global Variables
##########################################################################
N_FEATURES  = 1000
N_TOPICS    = 15
N_TOP_WORDS = 10

with open("noaa_data.json") as data_file:
    data = json.load(data_file)

stopwords = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", \
                "against", "all", "almost", "alone", "along", "already", "also","although",\
                "always","am", "among", "amongst", "amoungst", "amount",  "an", "and", "another", \
                "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  \
                "at", "back","be","became", "because","become","becomes", "becoming", "been", \
                "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", \
                "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", \
                "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", \
                "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", \
                "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", \
                "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", \
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
                "yours", "yourself", "yourselves", "the", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


##########################################################################
# Main Methods
##########################################################################
def createDict(nr_records=len(data)):
    """
    Creates a dictionary from the first n_records.
    Keys are record identifier names.
    Values are words from the description and keyword fields, and word frequencies (common stopwords are removed).
    """
    vectDict = {}
    for i in range(nr_records):
        name=data[i]['identifier']
        description=data[i]['description'].split()
        keywords= data[i]['keyword']
        keywordSep = []
        for k in keywords: keywordSep += k.split()
        allWords = map(lambda x:x.lower().strip(string.punctuation),(description+keywordSep))
        woStops = []
        for word in allWords:
            if word not in stopwords:
                woStops.append(word.encode('unicode-escape'))
        uniques = set(woStops)
        vectDict[name] = ' '.join(uniques)
    return vectDict


def createStops(nr_records=500):
    """
    Develop a suggested list of domain-specific stop words.
    Read through 500 records and take the 5 most common words from each entry.
    From this list, compile a list of unique words that appear in many entries.
    """
    wordList = []
    commonWords = []
    domainStops = []
    for i in range(nr_records):
        # pull the text out of the description and keyword fields
        description=data[i]['description'].split()
        keywordSep = []
        keywords= data[i]['keyword']
        for k in keywords: keywordSep += k.split()
        # make everything lowercase and strip out the punctuation
        allWords = map(lambda x:x.lower().strip(string.punctuation),(description+keywordSep))
        # build a list of the most common words for each record
        for word in allWords:
            if word not in stopwords:
                wordList.append(word.encode('unicode-escape'))
        commonWords.append([word[0] for word in (collections.Counter(wordList).most_common())[:10]])
    # return set of the most common words across entries
    for setwords in commonWords:
        for word in setwords:
            domainStops.append(word)
    return set(domainStops)


def mocksearch(searchterm,dictionary):
    """
    Test to see what percentage of the records each search word appears in
    """
    results = []
    for i in dictionary:
        if searchterm in dictionary[i]:
            results.append(i)
    presence = float(len(results))/len(dictionary)*100
    print "The term '%s' appears in %d percent of search results." % (searchterm, presence)
    return results


def topic_extraction(nr_records=len(data)):
    """
    Perform topic extraction on the first nr_records (defaults to all records).
    """
    # Extract the text from each record's description and keyword list
    fulllist = []
    for i in range(nr_records):
        shortlist = ' '.join(data[i]['keyword'])
        fulllist.append(shortlist+" "+data[i]['description'].encode('unicode-escape'))
    # Instantiate term frequency inverse document frequency model, using 1000 features & some normal english stop words
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=N_FEATURES, stop_words=stopwords)
    # Fit the model on the record text and transform each record into a sparse matrix
    tfidf = vectorizer.fit_transform(fulllist)
    # Create topic clusters for n topics, n here is 15
    nnmf  = NMF(n_components=N_TOPICS, random_state=1).fit(tfidf)
    names = vectorizer.get_feature_names()
    # For each topic, print out the 10 top words for each cluster
    for idx, topic in enumerate(nnmf.components_):
        print "Topic #{}:".format(idx+1)
        print (" ".join([names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]))
        print



if __name__ == '__main__':
    # # Create a dictionary where each record is stored in key-value pairs.
    # vect_dict = createDict()

    # # From that dictionary, compile a list of suggested domain-specific stop words.
    # domain_stops = createStops()

    # # For each word in the suggested stopwords, determine % of records it appears in
    # for word in domain_stops:
    #     mocksearch(word,vect_dict)

    # Perform topic extraction on all records
    topic_extraction()
