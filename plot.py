#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

import json
import requests
import sys, os
import itertools

#import python-Levenshtein for string matching
#http://pypi.python.org/pypi/python-Levenshtein/
#can be installed with "pip-2.7 install python-Levenshtein"
import Levenshtein
from Levenshtein import *

#encoding=utf8 
import sys
reload(sys)  
sys.setdefaultencoding('utf8')


plt.rcParams['figure.figsize'] = (12, 8)

# Function that computes prediction ration of username estimation 
# of given two social media account usernames
# Prediction is based on string matching via Levenshtein distance
# Input: DataFrame that has two columns, each column has the usernames
# for one social media account
# Metric is either Levenshtein or JaroWinkler
def predictionRatio(df, metric="Levenshtein"):
    #Generate all possible combinations for string matching
    soc_media_1, soc_media_2 = df.columns
    # Convert everything to lower case
    df[soc_media_1] = df[soc_media_1].str.lower()
    df[soc_media_2] = df[soc_media_2].str.lower()

    df_known = DataFrame([df[soc_media_1].tolist()] * df.shape[0], index=df.index, columns=df.index)
    df_search = DataFrame([df[soc_media_2].tolist()] * df.shape[0], index=df.index, columns=df.index)
    df_known_list = df_known.applymap(lambda x: list([x]))
    df_search_list = df_search.applymap(lambda x: list([x]))
    df_search_list = df_known_list+df_search_list.T

    # Find the indices of columns for each row  based on metric
    # For Levenshtein get the min., for JaroWinkler get the max.
    if metric == 'Levenshtein':
        search_res = df_search_list.applymap(lambda x: Levenshtein.distance(x[0], x[1]))
        indices = search_res.idxmin(axis=1)
    else:
        search_res = df_search_list.applymap(lambda x: Levenshtein.jaro_winkler(x[0], x[1]))
        indices = search_res.idxmax(axis=1)
    
    # Get the matches for social media account
    match = df[soc_media_2].ix[indices]
    df_t = DataFrame()
    df_t['actual'] = df[soc_media_2].reset_index(drop=True)
    df_t['match'] = match.reset_index(drop=True)
    # Find the ratio of correct matches
    match_count = (df_t.actual == df_t.match).value_counts()
    ratio = float(match_count[True]) / (match_count[True] + match_count[False])
    return ratio

def getSocialMediaMatchRatios(csv_file, metric='Levenshtein'):
    df = pd.read_csv(csv_file, encoding='utf-8')
    rel_cols = ['first_name', 'last_name', 'linkedin_username', 'facebook_username', 'twitter_username', 'instagram_username'] 
    username_cols = rel_cols[2:]

    combinations = list(itertools.combinations(username_cols, 2))
    username_pairs =  [[x, y] for x, y in combinations]

    ratios = {}
    for pair in username_pairs:
        test = df[pair].dropna(how='any')
        ratios[pair[0][:4]+'_'+pair[1][:4]] = predictionRatio(test, metric)

    return ratios

def plotSocialNetworkbyGender(male_usernames_file, female_usernames_file, metric='Levenshtein', fig_name='predict.png'):

    male_match_ratio =  getSocialMediaMatchRatios(male_usernames_file, metric)
    female_match_ratio = getSocialMediaMatchRatios(female_usernames_file, metric)

    num_els = len(male_match_ratio)
    male_match = male_match_ratio.values()
    female_match = female_match_ratio.values()
    tags = []
    tag_names = {'T':'Twitter', 'I': 'Instagram', 'F':'Facebook', 'L':'Linkedin'}
    for el in male_match_ratio.keys():
        splits = el.upper().split('_')
        s_m_1_key = splits[0][0]
        s_m_2_key = splits[1][0]
        tags.append(tag_names[s_m_1_key]+' vs. '+tag_names[s_m_2_key])

    # Convert to percentages
    male_match = [x*100 for x in male_match]
    female_match = [x*100 for x in female_match]

    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=11)
    index = np.arange(num_els)
    bar_width = 0.35
    opacity = 0.7
    rects1 = plt.bar(index, male_match, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Male')

    rects2 = plt.bar(index + bar_width, female_match, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Female')

    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Prediction Accuracy by Social Media Usernames and Gender')
    plt.xticks(index + bar_width, tags)
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)



def getUserInfoMatchRatios(df, metric='Levenshtein'):
    rel_cols = ['first_name', 'last_name', 'linkedin_username', 'facebook_username', 'twitter_username', 'instagram_username'] 
    username_cols = rel_cols[2:]
    userinfo_cols = rel_cols[:2]

    # Combine first and last name 
    df['first_last_name'] = df['first_name'] + df['last_name']
    userinfo_cols.append('first_last_name')

    combinations = [[y,x] for x in username_cols for y in userinfo_cols]

    ratios = {}
    for el in username_cols:
        ratios[el.split('_', 1)[0]] = []

    for pair in combinations:
        test = df[pair].dropna(how='any')
        info = pair[0]
        uname = pair[1].rsplit('_', 1)[0]
        ratios[uname].append((info, predictionRatio(test, metric)))
    return ratios



def plotSocialNetworkbyUserInfo(male_usernames_file, female_usernames_file,
                                metric='Levenshtein', fig_name='userinfo.png'):
    # Combine male and female data
    df_m = pd.read_csv(male_usernames_file, encoding='utf-8')
    df_f = pd.read_csv(female_usernames_file, encoding='utf-8')
    df_m.append(df_f)

    male_match_ratio = getUserInfoMatchRatios(df_m, metric)
    
    # Get match ratios per social media account
    linkedin = []
    facebook = []
    twitter = []
    instagram = []


    for key, val in male_match_ratio.iteritems():
        if key == 'linkedin':
            linkedin = [y*100.0 for x,y in val]
        elif key == 'facebook':
            facebook = [y*100.0 for x,y in val]
        elif key == 'twitter':
            twitter = [y*100.0 for x,y in val]
        elif key == 'instagram':
            instagram = [y*100.0 for x,y in val]
    
    # Get the label info
    tags = ['First Name Only', 'Last Name Only', 'First and Last Name']

    num_els = len(tags)
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=12)
    index = np.arange(num_els)*1.5
    bar_width = 0.25
    opacity = 0.7

    rects1 = plt.bar(index, linkedin, bar_width,
                     alpha=opacity,
                     color='k',
                     label='LinkedIn')

    rects2 = plt.bar(index+bar_width, facebook, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Facebook')
    rects3= plt.bar(index+2*bar_width, twitter, bar_width,
                     alpha=opacity,
                     color='dodgerblue',
                     label='Twitter')
    rects4= plt.bar(index+3*bar_width, instagram, bar_width,
                     alpha=opacity,
                     color='tan',
                     label='Instagram')       

    plt.ylabel('Prediction Accuracy (%)')
    plt.title('Predicting Social Media Username based on First and Last Name of Users')
    plt.xticks(index + 2*bar_width, tags)
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name)


plotSocialNetworkbyGender('males.csv', 'females.csv', fig_name='gen_social_leven.png')
plotSocialNetworkbyUserInfo('males.csv', 'females.csv', fig_name='username_first_last_name.png')
