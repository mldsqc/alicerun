import psycopg2
import pandas as pd
from datetime import date
import sys

import sqlite3
import numpy as np
from datetime import date
import configparser
import json
import requests
from base64 import b64encode



#TODO FIX the login credentials

""" Connect to Postgresql database.
    Reading TODO existing data from postgres  """

# db_config = read_db_config()
# conn = None
# try:
#     conn = psycopg2.connect("dbname='test' user='postgres' port='5432' host='localhost' password='Al24Ma26Boiko'")
#     # cursor = conn.cursor()
#
#     sql_query = pd.read_sql_query('''SELECT * FROM daily_answers''', conn)
#
#     df_daily_answers = pd.DataFrame(sql_query)
#
#     df_daily_answers.drop(df_daily_answers.columns.difference(['date_answer','time_answer', 'event','event_description'])
#                       , axis=1, inplace=True)
#
#     df_daily_answers['date_answer'] = df_daily_answers['date_answer'].astype('datetime64')
#     df_daily_answers['time_answer'] = df_daily_answers['time_answer'].astype('datetime64')
#
#     print (df_daily_answers)
#
# except psycopg2.OperationalError as error:
#     print("Shit happens   :(   \n\n  ")
#     print(error)
# finally:
#     if conn is not None:
#         conn.close()
#         print('Connection closed  ')


def load_test_emotions_db():
    # parsing DBs paths in file system
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_db_file2 = config.get('DB_PATHS', 'DB_EMOTIONS_TEST')
    emotions_test = pd.read_csv(test_db_file2)
    return emotions_test

# load_test_emotions_db()



def load_emotions_habits_values():
    emotions_values = {
    'burnouted':-2,
    'emotionally balanced': 2,
    'emotionally UNbalanced': -2,
    'pleasure from done tasks': 2,
    'UNPREDICTABLE EMOTIONAL WOWs': 2,
    'motivated': 1,
    'focused': 1,
    'procrastinated': -2,
    'productive': 1,
    'admiration': 1,
    'adoration': 1,
    'aesthetic': 1,
    'appreciation': 2,
    'amusement': 1,
    'anger': -1,
    'anxiety': -2,
    'awe': 1,
    'awkwardness': -1,
    'boredom': -1,
    'calmness': 2,
    'confusion': -2,
    'craving': 1,
    'disgust': -1,
    'empathic': 2,
    'pain': -1,
    'entrancement': 1,
    'excitement': 2,
    '+fear': 2,
    '-fear': -2,
    'horror': -2,
    'interest': 1,
    'joy': 1,
    'nostalgia': 1,
    'relief': 2,
    'romance': 1,
    'sadness': -1,
    'satisfaction': 2,
    'sexual desire': 1,
    'surprise': 2,
    'pride': 2,
    'happiness': 2,
    'inspiration': 1,
    'fascination': 1 ,
    '-surprise': -2,
    '+surprise': 2,
    'dissatisfaction': -1,
    'embarrassed': -1,
    'indignation': -1
}

    habits_values = {

'PHYSICAL': {
    'slept_GOOD': 2,
    'cold shower': 3,
    '10 pushups everyday': 3,
    'big physical activity': 3,
    'over_eated': -2,
    'fastfood': -1,
    'cafe': 1
},
'FUN_RECREATION': {

    'traveled': 1,
    'drugs': 2,
    'jrk': 1,
    'porn': -1,
    'too much movies': -2,
    'too much youtube': -2,
    'too much social media': -2,
    'too much news': -2,
    'any pain': -2,
    '+youtubed': 1,
    'film': 1,
    'tvshow': 1,
    'cinema': 1,
    'gaming': 1
},
'INTELECTUAL': {

    'made smth for selfefficiency': 3,
    'how many info i consumed vs generated out in the world': 3,
    'running on plans feeling': 2,
    'psycho practices': 2,
    '5 minute journal': 2,
    'meditation': 2,
    'look inside 4 feelings on whole life': 2,
    'bucket list': 5,
    'wish list': 5,
    'reading': 1,
    'studing': 1,
    'chess': 1,
    'ankicards': 2,
    'languages': 1
},
'LOVE ROMANCE SEX': {

    'sex': 2,
    'new sex partner': 3,
    'new sex practices': 2,
},
'PARTNER': {
    'harmony_pleasurefull': 2,
    'critiqued_by_HER': -1,
    'critiqued_by_ME': -1,
    'arguing': -1,
    'work thru complicated situations': 2,
    'common goals completion': 2
},
'SOCIAL FRIENDS': {
    'social offline': 1,
    'new people': 1,
    'old_friends': 2
},
'FINANCIAL': {
    'encreased income':5,
    'financial reduce costs':5,
    'investing':5
}
}

    emotions_and_habits_values = {
        'burnouted': -2,
        'emotionally balanced': 2,
        'emotionally UNbalanced': -2,
        'pleasure from done tasks': 2,
        'UNPREDICTABLE EMOTIONAL WOWs': 2,
        'motivated': 1,
        'focused': 1,
        'procrastinated': -2,
        'productive': 1,
        'admiration': 1,
        'adoration': 1,
        'aesthetic': 1,
        'appreciation': 2,
        'amusement': 1,
        'anger': -1,
        'anxiety': -2,
        'awe': 1,
        'awkwardness': -1,
        'boredom': -1,
        'calmness': 2,
        'confusion': -2,
        'craving': 1,
        'disgust': -1,
        'empathic': 2,
        'pain': -1,
        'entrancement': 1,
        'excitement': 2,
        '+fear': 2,
        '-fear': -2,
        'horror': -2,
        'interest': 1,
        'joy': 1,
        'nostalgia': 1,
        'relief': 2,
        'romance': 1,
        'sadness': -1,
        'satisfaction': 2,
        'sexual desire': 1,
        'surprise': 2,
        'pride': 2,
        'happiness': 2,
        'inspiration': 1,
        'fascination': 1,
        '-surprise': -2,
        '+surprise': 2,
        'dissatisfaction': -1,
        'embarrassed': -1,
        'indignation': -1,
        'slept_GOOD': 2,
        'cold shower': 3,
        '10 pushups everyday': 3,
        'big physical activity': 3,
        'over_eated': -2,
        'fastfood': -1,
        'cafe': 1,
        'traveled': 1,
        'drugs': 2,
        'jrk': 1,
        'porn': -1,
        'too much movies': -2,
        'too much youtube': -2,
        'too much social media': -2,
        'too much news': -2,
        'any pain': -2,
        '+youtubed': 1,
        'film': 1,
        'tvshow': 1,
        'cinema': 1,
        'gaming': 1,
        'made smth for selfefficiency': 3,
        'how many info i consumed vs generated out in the world': 3,
        'running on plans feeling': 2,
        'psycho practices': 2,
        '5 minute journal': 2,
        'meditation': 2,
        'look inside 4 feelings on whole life': 2,
        'bucket list': 5,
        'wish list': 5,
        'reading': 1,
        'studing': 1,
        'chess': 1,
        'ankicards': 2,
        'languages': 1,
        'sex': 2,
        'new sex partner': 3,
        'new sex practices': 2,
        'harmony_pleasurefull': 2,
        'critiqued_by_HER': -1,
        'critiqued_by_ME': -1,
        'arguing': -1,
        'work thru complicated situations': 2,
        'common goals completion': 2.,
        'social offline': 1,
        'new people': 1,
        'old_friends': 2,
        'encreased income': 5,
        'financial reduce costs': 5,
        'investing': 5
}
    return emotions_values, habits_values, emotions_and_habits_values


# EMOTIONAL
def count_emotion_perday(df, emotion):
    """ certain emotion amount per day"""

    return df.groupby([df['event_description'] == emotion, pd.Grouper(key='date_answer', axis=0,
                                                                                         freq='D')]).count()

def emotional_balance_for_testdb(df, emotions_values=load_emotions_habits_values()[0]):
    """ TODAY EMOTIONAL BALANCE
     !!!! in this DF not only emotions but HABITS TOO"""

    daily_list_registered = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])['event_description'])

    daily_emotional_balance = 0
    for i in emotions_values.keys():
        if i in daily_list_registered.values[1, 1].tolist():
            daily_emotional_balance += emotions_values[i]

    return daily_emotional_balance


def habital_weekly_balance_for_testdb(df, emotions_values=load_emotions_habits_values()[1]):
    """ TODAY EMOTIONAL BALANCE
     !!!! in this DF not only emotions but HABITS TOO"""

    daily_list_registered = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])['event_description'])

    daily_emotional_balance = 0
    for i in emotions_values.keys():
        if i in daily_list_registered.values[1, 1].tolist():
            daily_emotional_balance += emotions_values[i]

    return daily_emotional_balance


def procrastination_frustration_balance_for_testdb(df, emotions_values=load_emotions_habits_values()[0]):
    """ Counting frustration procrastination based on """


    daily_list_registered = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])['event_description'])

    daily_emotional_balance = 0
    for i in emotions_values.keys():
        if i in daily_list_registered.values[1, 1].tolist():
            daily_emotional_balance += emotions_values[i]

    return daily_emotional_balance


def emotional_balance(df, emotions_values=load_emotions_habits_values()[0]):
    """ TODAY EMOTIONAL BALANCE
     !!!! in this DF not only emotions but HABITS TOO"""


    daily_list_registered = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])['event_description'])

    daily_emotional_balance = 0
    for i in emotions_values.keys():
        if i in daily_list_registered.values[1, 1].tolist():
            daily_emotional_balance += emotions_values[i]

    return daily_emotional_balance



def area_balance(df, dict_values):
    """TODAY area balance . input dictionary"""

    balance = 0
    daily_list_registered = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])[
                                             'event_description'])
    for i in dict_values.keys():
        if i in daily_list_registered.values[1,1].tolist():
            balance += dict_values[i]

    return balance

# area_balance(emotions_values)


def daily_emotion_list(df):
    daily_list_emotions = pd.DataFrame(df.groupby([df['date_answer'] == date.today().strftime("%Y/%m/%d")])[
                                           'event_description'])
    return daily_list_emotions


# def areas_dayly_progress(df):
#    ''' # AREAS PROGRESS BY
#     # TODO write to DB each day progress
#     # TODO MAKE timeseries for progress'''
#
#     progress = ''
#     # habits_areas_list = ['SOCIAL FRIENDS', 'PARTNER', 'LOVE ROMANCE SEX', 'INTELECTUAL', 'FUN_RECREATION', 'PHYSICAL']
#     for area, habits in habits_values.items():
#         # print(area, ' -----   ', habits)
#         progress +=  str(area_balance(df_daily_answers, habits)) + ' -----   ' + str(area) + \n
#
#     return progress







