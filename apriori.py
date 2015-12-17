# Sai J

# Association rule Mining and Recommendation System based on Sales data

from operator import mul

from __future__ import division
from itertools import combinations

import numpy as np
import scipy as sc
import pandas as pd

# Develop code for Apriori Algo
# Input = Pandas DataFrame with Columns as items whose associaton rules need to be identified
# & rows as individual transactions. Values are binary: 0s and 1s.
# 0: Non-occurrence
# 1: Occurrence
# No NaNs or Nulls

# Takes a list of iterable values (columns or combination of columns) and returns next set of combinations of higher length
# data = Pandas Data Frame whose columns are items whose association rules need to be identified and rows are transactions
# support = Minimum frequency a column/association must have
# lift_base = If support not supplied; Frequency of occurrence of an association/ Expected co-occurrence (assuming mutual independence)
def filter_support(iterable, data, support= None, lift_base= None):
    if (((support is None) & (lift_base is None)) | ((support is not None) & (lift_base is not None))):
        print 'Supply proper Support or Lift_base'
        return None
    passed_threshold = []
    len_data = len(data)
    if type(iterable[0])==list:
        combined_list = list(set([y for x in iterable for y in x]))
        len_list = len(iterable[0])
    else:
        combined_list = iterable
        len_list = 1
    list_comb = combinations(combined_list, len_list+1)
    for item in list_comb:
        cols = list(item) # Convert into list since combinations returns tuples
        assoc_prob = sum(data[cols].product(axis=1))/len_data # Get proportion of the combination where all columns are 1s
        if (support is not None) & (assoc_prob>= support): # If absolute support is known
            passed_threshold.append(cols)
            continue
        orig_prob = reduce(mul,[sum(data[col])/len_data for col in cols],1) # Probabilistic Likelihood of association
        if (lift_base is not None) & (orig_prob!=0) & ((assoc_prob/orig_prob)>=lift_base):
            passed_threshold.append(cols)
            continue
    return passed_threshold

# Call filter_support for different lengths of associations
# Returns association rules that qualify
def get_assoc_list(max_assoc_len, init_col_list, data, support= None, lift_base = None):
    len_list = 1
    next_iter = init_col_list
    final_assoc_list = []

    while (len_list < max_assoc_len) & (len(next_iter)>1):
        next_iter = filter_support(next_iter, data, None, lift_base)
        final_assoc_list = final_assoc_list + next_iter
        if len(next_iter)!=0:
            len_list = len(next_iter[0])
        else:
            break
    return final_assoc_list
