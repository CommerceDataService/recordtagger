#!/usr/bin/python
# tag_filter.py
#
#
# Title:        Tag Recommendation Filtering
# Author:       Rebecca Bilbro
# Version:      1.0
# Date:         2/25/16
# Organization: Commerce Data Service, U.S. Department of Commerce


#####################################################################
# Imports
#####################################################################
from __future__ import print_function  # Not necessary for Python 3
import csv

#####################################################################
# Tag filtration
#####################################################################
def tagfilter(infile,outfile):
    with open(outfile, 'wb') as outcsvfile:
        writer = csv.writer(outcsvfile)
        writer.writerow(["record_index","suggested_keywords"])

        with open(infile, 'rb') as incsvfile:
            reader = csv.reader(incsvfile, delimiter=',')
            next(reader, None)
            for row in reader:
                index = row[0]
                oldtext = row[1].split()
                allwords = row[3].split()
                filtered = [x for x in set(allwords) if x not in oldtext]
                writer.writerow([index,filtered])

if __name__ == '__main__':
    messytags = 'records_to_ldaclusters_v2.csv'
    prettytags = 'lda_tag_recommendations.csv'
    tagfilter(messytags,prettytags)
