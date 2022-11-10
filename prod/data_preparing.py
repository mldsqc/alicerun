import sys

import pandas as pd
import sqlite3
import numpy as np
from datetime import date
import configparser
import json
import requests
from base64 import b64encode
import datetime

from bot_answers_analysis import load_emotions_habits_values

# parsing DBs paths in file system
config = configparser.ConfigParser()
config.read('config.ini')


# config.read('/src/prod/config.ini')


def sessions_download():
    """
    problem with api - not giving more than 30? entries. so concatenating with archive file
    importing sessions from TOGGL API
    +
    concatenating with sessions archive due to toggl api limits with 50 recent time entries

    #TODO SESSIONS INFO. NUMBER SESSIONS PER TASK,DIFFICULTY VS TTC VS NUMBER OF SESSIONS, IMPORT SESSIONS TO GOOGLE CALENDAR

    get all sessions archive by dates
    curl -v -u TOKEN:api_token -X GET "https://api.track.toggl.com/api/v8/time_entries?start_date=2013-03-10T15%3A42%3A46%2B02%3A00&end_date=2023-03-12T15%3A42%3A46%2B02%3A00"
    """

    def datetime_convertion_colunm(df, column):
        df[column] = pd.to_datetime(df[column])

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    # sessions_archive_file2 = config.get('toggl', 'sessions_archive2')
    session_archive = pd.read_csv("./android_db/Toggl_time_entries3.csv")

    # authHeader = config.get('toggl', 'TOKEN') + ":" + "api_token"
    # data = requests.get('https://api.track.toggl.com/api/v9/me/time_entries',
    #                     headers={'content-type': 'application/json', 'Authorization': 'Basic %s' % b64encode(authHeader
    #                                                                                                          .encode()).decode(
    #                         "ascii")})
    #
    # new_sessions = pd.read_json(json.dumps(data.json()), orient='records')
    # # new_sessions = new_sessions[1:,:]
    #
    # intersection_col_list = intersection(session_archive.columns, new_sessions.columns)
    # intersection_col_list.remove('billable')
    #
    # session_archive = session_archive[session_archive.columns.intersection(intersection_col_list)]
    # new_sessions = new_sessions[new_sessions.columns.intersection(intersection_col_list)]
    #
    # datetime_convertion_colunm(new_sessions, column='start')
    # datetime_convertion_colunm(new_sessions, column='stop')
    # datetime_convertion_colunm(new_sessions, column='at')
    datetime_convertion_colunm(session_archive, column='start')
    datetime_convertion_colunm(session_archive, column='stop')
    datetime_convertion_colunm(session_archive, column='at')

    # sessions = pd.concat([session_archive, new_sessions])
    # sessions = sessions.drop_duplicates()
    return session_archive


def read_db_paths():
    """reading synced mobile device DBs paths"""
    DB_ADASH1 = config.get('DB_PATHS', 'DB_ADASH1')
    DB_ADASH2 = config.get('DB_PATHS', 'DB_ADASH2')
    DB_ADASH3 = config.get('DB_PATHS', 'DB_ADASH3')
    DB_ADASH4 = config.get('DB_PATHS', 'DB_ADASH4')
    DB_TODO = config.get('DB_PATHS', 'DB_TODO')
    DB_ACTWATCH = config.get('DB_PATHS', 'DB_ACTWATCH')
    DB_GADGETBRIDGE = config.get('DB_PATHS', 'DB_GADGETBRIDGE')
    DB_EMOTIONS_TEST = config.get('DB_PATHS', 'DB_EMOTIONS_TEST')

    return DB_ADASH1, DB_TODO, DB_ACTWATCH, DB_GADGETBRIDGE, DB_ADASH2, DB_ADASH3, DB_ADASH4, DB_EMOTIONS_TEST


def read_sql(DB, query):
    conn = sqlite3.connect(DB)

    try:
        sql_query = pd.read_sql_query(query, conn)
        df = pd.DataFrame(sql_query)
        # print(df)
        return df

    finally:
        if conn is not None:
            conn.close()
            print('Connection closed  ')


# TODO write_df_todb
def write_df_todb(DB, query):
    conn = sqlite3.connect(DB)

    try:
        sql_query = pd.read_sql_query(query, conn)
        df = pd.DataFrame(sql_query)
        # print(df)
        return df

    finally:
        if conn is not None:
            conn.close()
            print('Connection closed  ')
    pass


def df_tasks_prepare(DB_TODO):
    """ DB - from microsoft TO-DO
        read sql DB for tasks names, read metrics
        producing dataframe for task oriented metrics"""

    df_tasks = read_sql(DB_TODO, '''SELECT
                         subject, original_body_content, task_folder_local_id, status, importance, original_body_content,
                         body_last_modified, created_datetime, completed_datetime
                         FROM tasks
                      ''')

    df_group = read_sql(DB_TODO, '''SELECT
                         group_local_id, name,local_id
                         FROM task_folders
                      ''')
    df_group.rename(columns={'group_local_id': 'task_folder_local_id', 'name': 'group'}, inplace=True)

    # print(df_group)
    df_areas = read_sql(DB_TODO, '''SELECT
                         local_id, name
                         FROM groups
                      ''')
    df_areas.rename(columns={'local_id': 'task_folder_local_id', 'name': 'life_area'}, inplace=True)

    df_joined = df_group.join(df_areas.set_index('task_folder_local_id'), on='task_folder_local_id', how='left')
    df_joined.drop('task_folder_local_id', inplace=True, axis=1)
    df_joined.rename(columns={'local_id': 'task_folder_local_id'}, inplace=True)

    # print(df_joined)
    return df_tasks, df_joined


def emotions_habits_df_prepare():
    """preparing tracked emotions habits data for being plotted
    #data is from archived test dataset, !!!!!!NOT!!!!!!!! from postgresDB

    # for PLOTTING ANIMATED TIMESERIES FOR EMOTIONS ARCHIVE df_emotions1 is used

    # one-hot encoding for trying get data for plotting timeseries on radial plot
    # getting dummies from list from grouped by rounded 10m datetime periods
    # https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies

    """
    global emotions_values, habits_values, emotions_and_habits_values


    def floor_dt(dt, interval=10):
        """rounding datetime column to 10 min intervals"""

        replace = (dt.minute // interval) * interval
        return dt.replace(minute=replace, second=0, microsecond=0)

    # DB_EMOTIONS_TEST = read_db_paths()[-1]
    emotions_habits = pd.read_csv('./android_db/habits_emotions.csv')
    emotions_values, habits_values, emotions_and_habits_values = load_emotions_habits_values()
    # emotions_habits

    emotions_habits['datetime'] = emotions_habits.date.astype('str') + ' ' + emotions_habits.time.astype('str')
    emotions_habits['datetime'] = emotions_habits['datetime'].astype('datetime64')

    emotions_habits.drop(['date', 'time'], axis=1, inplace=True)
    emotions_habits['datetime_rnd'] = emotions_habits.datetime.apply(floor_dt)
    emotions_habits['emo_hab'] = np.where(emotions_habits['act'].isin(emotions_values.keys()), 0, 1)

    # emotions_habits = emotions_habits.set_index('datetime_rnd')

    # making different dfs for emotions and habbits
    df_emotions = emotions_habits[emotions_habits.emo_hab == 0]
    df_habits = emotions_habits[emotions_habits.emo_hab == 1]

    df_emotions = df_emotions.reset_index()

    df_emotions1 = df_emotions.groupby('datetime_rnd')['act'].agg(list).reset_index()  # .iloc[1,:]

    # GENERATION getdummies from list of emotions (act column) grouped by 10min periods
    df_emotions1 = df_emotions1.join(df_emotions1['act'].str.join('|').str.get_dummies())
    # df_emotions1

    return df_emotions, df_habits, df_emotions1


##FUNCTIONAL THINGS
def first_session_date_by_taskname(df_sessions, taskname):
    """return date of first session of task

    """
    if len(df_sessions[df_sessions.subject == taskname].start.agg(list)) == 0:
        return None
    else:
        return min(df_sessions[df_sessions.subject == taskname].start.agg(list))


# first_session_date_by_taskname('calendar sync visualisation 4 2 3 4 1')
# df_sessions[df_sessions.subject=='calendar sync visualisation 4 2 3 4 1'].head()

# load_emotions_habits_values()

def date_of_task_completion(df_tasks_3, taskname):
    """take date of completed task"""
    mask__ = (df_tasks_3.subject == taskname)
    # &(isinstance(df_tasks_3[df_tasks_3.subject==taskname].completed_datetime,datetime.datetime))
    return df_tasks_3[mask__].completed_datetime


# date_of_task_completion('test forked running bot').astype('datetime64') #check


def common_list_items(a, b):
    from collections import Counter
    ca = Counter(a)
    cb = Counter(b)

    result = [a for b in ([key] * min(ca[key], cb[key])
                          for key in ca
                          if key in cb) for a in b]
    return result


# TODO create mean of emotion balance during ALL task sessions
def count_emotion_balance_of_task_completion_day_by_task_name(df_tasks_3, df_emotions1, taskname, regime=0,
                                                              values_list=[]):
    """get list of emotions during !!!! COMPLETION_day!!!!! by taskname
        part for counting frustration and procrastination when regime == 1
    """

    global emotions_values, habits_values, emotions_and_habits_values

    dateee = date_of_task_completion(df_tasks_3, taskname).iloc[0]
    # print(dateee)
    # print(df_emotions1['datetime_rnd'].dt.strftime("%Y-%m-%d"))
    mask___ = (df_emotions1['datetime_rnd'].dt.strftime("%Y-%m-%d") == dateee)
    # print(mask___)
    daily_list_of_emotions = df_emotions1[mask___].act.sum()
    # print(dayly_list_of_emotions)
    daily_emotional_balance = 0

    # not all tasks are completed after i started tracking emotions,
    # so check if 0 registered emotions
    if daily_list_of_emotions != 0 and regime == 0:
        for ik in emotions_values.keys():
            if ik in daily_list_of_emotions:
                daily_emotional_balance += emotions_values[ik]
            return daily_emotional_balance
    # part for counting frustration and procrastination
    else:
        if daily_list_of_emotions != 0 and regime == 1:
            for ik in common_list_items(emotions_and_habits_values.keys(), values_list):
                if ik in daily_list_of_emotions:
                    daily_emotional_balance += emotions_and_habits_values[ik]
            return daily_emotional_balance
    # print(daily_emotional_balance)


##PRODUCING FEATURES
def tasks_read_metrics(df, df_joined):
    """ joining all data about TO-DO lists groups and tasks and making
    standalone columns for metrics"""

    df['TTC'], df['PRI'], df['DIFF'], df['PLEAS'], df['RESIS'] = None, None, None, None, None
    df['metricks'] = df['subject'].str[-9:].str.replace(" ", "").str.extract(r'(\d+[.\d]*)')
    # df['group'], df['life_area'] = None, None
    df['TTC'] = np.where(df.metricks.str.isnumeric(), df.metricks.str[0], df.TTC)
    df['PRI'] = np.where(df.metricks.str.isnumeric(), df.metricks.str[1], df.PRI)
    df['DIFF'] = np.where(df.metricks.str.isnumeric(), df.metricks.str[2], df.DIFF)
    df['PLEAS'] = np.where(df.metricks.str.isnumeric(), df.metricks.str[3], df.PLEAS)
    df['RESIS'] = np.where(df.metricks.str.isnumeric(), df.metricks.str[4], df.RESIS)
    # cols = 'task_folder_local_id'
    df2 = df.join(df_joined.set_index('task_folder_local_id'), on='task_folder_local_id', how='left')
    # df3=df2.join(df_areas.set_index(cols), on=cols)
    # df3=df.merge(df_areas, how='left')

    # print(df2['name'].isna)
    # expected_result = pd.merge(df, df_areas, on = 'task_folder_local_id', how = 'left')
    # df3=df.join(df_areas, on=['task_folder_local_id'])
    # df_tasks = df2
    # print(df2)
    return df2


# TODO PROBLEM 2nd and 3rd parts overwrites previous values
def tasks_motivation(df):
    """COUNT MOTIVATION PER TASK
    count MOTIVATION metric for each task where there are metrics"""
    if 'motivation' not in df.columns:
        df['motivation'] = None  # TODO MUST be done once WHEN ITS NOT WRITTEN TO DB YET

    #which are notstarted
    df['motivation'] = np.where(((df['status'] == 'NotStarted') & (df['metricks'] is not None) & (df['motivation'] is
                                                                                                  None)),
                                (df['PLEAS'].astype('Int32') * df['DIFF'].astype('Int32') * df['PRI'].astype('Int32')) /
                                (1 + (df['TTC'].astype('Int32') * df['RESIS'].astype('Int32') *
                                      (df['created_datetime'].astype('datetime64').astype(np.int64) /
                                       ((df['created_datetime'].astype('datetime64') - pd.DateOffset(months=1))
                                        .astype(np.int64)))
                                      )),
                                0)
    #which are completed
    df['motivation'] = np.where(
        ((df['motivation'] == 0) & (df['status'] == 'Completed') & (df['metricks'] is not None)),
        (df['PLEAS'].astype('Int32') * df['DIFF'].astype('Int32') * df['PRI'].astype(
            'Int32')) / (1 + (df['TTC'].astype('Int32') * df['RESIS'].astype('Int32') *
                              (df['created_datetime'].astype('datetime64').astype(np.int64) /
                               ((df['body_last_modified'].astype('datetime64') - df['completed_datetime'].astype(
                                   'datetime64'))
                                .astype(np.int64)))
                              )), None)

    # print(df)
    return df


def after_metric_ttc(df, df1):
    """# ERROR RATE NOT MORE THAN 15% FOR TTC METRIC FOR EACH TASK IN DF_TASKS
    # (FOR EXAMPLE TIME TO COMPLETION SHOULD BE PREDICTED IN 85% accuracy)
    # add to df_tasks column for summary time
    df=df_sessions, df1=df_tasks

    making df_tasks['duration']

    """

    df_sessions_tasks_duration = df.groupby(['subject'])['duration'].sum()
    # df_sessions_tasks_duration.rename(columns={'local_id': 'task_folder_local_id'}, inplace=True)
    # df_sessions_tasks_duration.rename(columns={'description': 'subject'}, inplace=True)
    df1 = pd.merge(df1, df_sessions_tasks_duration.reset_index(), left_on='subject', right_on='subject', how='left')

    # df1 = df1.join(df_sessions_tasks_duration, on='subject', how='left')
    # df1['TTC_aft'] = (df1['duration'].astype('Int32') / 3600) # error converting float to integer
    df1['TTC_aft'] = (df1['duration'] / 3600)
    return df1


def check_error_rate_predicting_ttc(df1):
    """# ERROR RATE NOT MORE THAN 15% FOR TTC METRIC FOR EACH TASK IN DF_TASKS
    # (FOR EXAMPLE TIME TO COMPLETION SHOULD BE PREDICTED IN 85% accuracy)
    # add to df_tasks column for summary time
    df=df_sessions, df1=df_tasks

    making df_tasks['duration']

    """
    df1['underrated'] = (df1['duration'] / 3600 - df1['TTC'].astype('Int32')) > (0.15 * df1['TTC'].astype('Int32') + 0.01)
    return df1


# ADDING MORE FEATURES for assuming metrics of tasks, based on secondary features
def after_features_assumpted(df, df1, df_tasks_3, df_emotions1):
    """   to input df=df_sessions, df1=df_tasks_3
       TODO, or mean of all days when were sessions of that task)
           # FOR that ONE  is NEEDED df_emotions1 and df_tasks_3 DFs
           # end day emotional balance (
        TODO, or mean of all days when were sessions of that task)

       # TTC_aft - sessions datetime sum if task done
       # PRI_aft - compared date started(first session in toggl)-created /time done for each task compare
       # PLEASURE_aft - end day emotional balance
       # DIFCLT_aft -  select frustration type of emotions, return were they there,
                         input list of sessions dates of certain
                         list of emotions pleasuring or stress  for certain task done

       # RES_aft amount of procrastination, too much youtube, or mobile screentime(????)
       #underrated as for TTC metric
       """

    df_tasks_3_t = after_metric_ttc(df, df1)
    # df_tasks_3_t[df_tasks_3_t.metricks.notnull()] #checking

    df_tasks_3_t['completed_datetime'] = df_tasks_3_t['completed_datetime'].astype('datetime64')
    df_tasks_3_t['created_datetime'] = df_tasks_3_t['created_datetime'].astype('datetime64')

    # TODO check ! why there is so small amount of counted cells
    for i in df_tasks_3_t[df_tasks_3_t.status == 'Completed'].subject:
        if isinstance(first_session_date_by_taskname(df_sessions=df, taskname=i), datetime.datetime):
            mask_ = (df_tasks_3_t.subject == i)
            df_tasks_3_t['PRI_aft'] = (df_tasks_3_t[mask_].completed_datetime -
                                       first_session_date_by_taskname(df_sessions=df, taskname=i).replace(
                                           tzinfo=None)) / (
                                              df_tasks_3_t[mask_].created_datetime - first_session_date_by_taskname(
                                          df_sessions=df, taskname=i).replace(tzinfo=None))

    # emotion balance by date of completed task
    for k in df_tasks_3_t[df_tasks_3_t.status == 'Completed'].subject:
        # print(k)
        indd = df_tasks_3_t[df_tasks_3_t.subject == k].index
        df_tasks_3_t.loc[indd, 'PLEASURE_aft'] = count_emotion_balance_of_task_completion_day_by_task_name(df_tasks_3,
                                                                                                           df_emotions1,
                                                                                                           k, regime=0)
    # df_tasks_3_t[df_tasks_3_t.status=='Completed'] #cheking

    # make lists for frustration and procrastination
    frustration = ['burnouted', 'emotionally UNbalanced', 'anxiety', 'confusion', '-fear', 'sadness', '-surprise',
                   'dissatisfaction']
    procrastination = ['burnouted', 'procrastinated', 'boredom', 'horror', 'jrk', 'porn', 'too much movies',
                       'too much youtube', 'too much social media', 'too much news']

    # TODO, when data on completed tasks will be more, what happening with that columns
    # count daily amount of such emotions
    for k in df_tasks_3_t[df_tasks_3_t.status == 'Completed'].subject:
        # print(k)
        indd = df_tasks_3_t[df_tasks_3_t.subject == k].index
        df_tasks_3_t.loc[indd, 'DIFCLT_aft'] = count_emotion_balance_of_task_completion_day_by_task_name(df_tasks_3,
                                                                                                         df_emotions1,
                                                                                                         k, regime=1,
                                                                                                         values_list=frustration)
        df_tasks_3_t.loc[indd, 'RES_aft'] = count_emotion_balance_of_task_completion_day_by_task_name(df_tasks_3,
                                                                                                      df_emotions1,
                                                                                                      k, regime=1,
                                                                                                      values_list=procrastination)

    # df_tasks_3_t[df_tasks_3_t.DIFCLT_aft.notna()] #check

    check_error_rate_predicting_ttc(df1=df_tasks_3_t)
    return df_tasks_3_t


def number_tasks_per_day(df, date=date.today().strftime("%Y-%m-%d")):
    """NUMBER OF completed TASKS PER DAY - TODAY DEFAULT DAY"""
    return df.groupby(['completed_datetime']).count()
    # [date]


def number_tasks_per_day_per_task(df, date=date.today().strftime("%Y-%m-%d")):
    '''NUMBER OF TASKS PER DAY - TODAY DEFAULT DAY'''
    return df.groupby(['completed_datetime', 'life_area', 'group']).count().iloc[:, :1]
    # .get_group('MY CV REVIEW')


# number_tasks_per_day(2022-7-19)
# number_tasks_per_day()


# df_tasks.groupby(['completed_datetime']).count()


def number_hard_tasks_per_m(df):
    '''TASKS done with difficulty - number of done difficult tasks per month
        to do >4'''

    return df.groupby(['completed_datetime', 'DIFF']).count()  # .drop(['DIFF':], axis=1, inplace=True)  #
    # .get_group('MY
    # CV REVIEW')


# number_tasks_per_day(2022-7-19)
# number_hard_tasks_per_m()


def check_goals_difficulty(df):
    """TODO add timeperiod
    goals difficulty should be in the middle rate, not over max rate a
        TO DO ADD PER MONTH
         and dont correlate with burnout
        # df_tasks.groupby(['completed_datetime', 'DIFF']).count()"""

    return df['DIFF'].astype('Int32').mean() > 4 #.rolling(30)


def tasks_done_per_month(df):
    """ TASKS done per month

    # df_tasks.groupby(['completed_datetime']).count()
    # df_tasks.query('completed_datetime').head()"""

    df['completed_datetime'] = df['completed_datetime'].astype('datetime64')
    return df.groupby(pd.Grouper(key='completed_datetime', axis=0,
                                 freq='M')).count()


def amount_tasks_done_life_areas(df, time_period='M'):
    """# AMOUNT OF tasks done in life areas in month"""

    df['completed_datetime'] = df['completed_datetime'].astype('datetime64')
    # (['life_area']).count()
    return df.groupby(['life_area', pd.Grouper(key='completed_datetime', axis=0,
                                               freq=time_period)]).count()


def amount_complited_tasks_permonth(df):
    """NUMBER OF COMPLETED TASKS FROM LISTS PER MONTH"""
    return df.groupby([df['status'] == 'Completed', pd.Grouper(key='completed_datetime', axis=0,
                                                               freq='M')]).count()


def amount_complited_tasks_per_list_month(df, group_name):
    ''' bucketlist done tasks per month'''
    return df.groupby([df['group'] == group_name, pd.Grouper(key='completed_datetime', axis=0,
                                                             freq='M')]).count()


def amount_complited_tasks_per_area_week(df, group_name):
    ''' bucketlist done tasks per week'''
    return df.groupby([df['life_area'] == group_name, pd.Grouper(key='completed_datetime', axis=0, freq='W')]).count()


def amount_new_tasks_per_day(df):
    """# NUMBER OF NEW TASKS PER DAY"""

    df['created_datetime'] = df['created_datetime'].astype('datetime64')

    number_new_tasks_pday = df.groupby(pd.Grouper(key='created_datetime', axis=0,
                                                  freq='D')).count().reset_index()
    return number_new_tasks_pday.drop(number_new_tasks_pday.columns.difference(['created_datetime', 'subject']), axis=1)


def done_tasks_and__creativity_simple_metricks(df_tasks_3_t):
    ######## compared amount of done tasks in this month to mean amount
    dff = tasks_done_per_month(df_tasks_3_t).iloc[:,:1]
    mean_amount_done_tasks_pmonth = dff.reset_index().subject.mean()
    last_month_amount_done_tasks = dff.reset_index().subject.iloc[-1]
    how_changed = int(100*(last_month_amount_done_tasks/mean_amount_done_tasks_pmonth)-100)

    #creativity as addded tasks to to do lists
    creat = amount_new_tasks_per_day(df_tasks_3_t)
    mean_creat = creat.subject.mean()
    today_creat = creat.subject.iloc[-1]
    how_changd_2 = int(100*(today_creat/mean_creat)-100)

    return last_month_amount_done_tasks, how_changed, today_creat, how_changd_2



####Checking RAM USAGE
def obj_size_fmt(num):
    if num < 10 ** 3:
        return "{:.2f}{}".format(num, "B")
    elif (num >= 10 ** 3) & (num < 10 ** 6):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 3), "KB")
    elif (num >= 10 ** 6) & (num < 10 ** 9):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 6), "MB")
    else:
        return "{:.2f}{}".format(num / (1.024 * 10 ** 9), "GB")


def memory_usage():
    memory_usage_by_variable = pd.DataFrame({k: sys.getsizeof(v) for (k, v) in globals().items()}, index=['Size'])
    memory_usage_by_variable = memory_usage_by_variable.T
    memory_usage_by_variable = memory_usage_by_variable.sort_values(by='Size', ascending=False).head(10)
    memory_usage_by_variable['Size'] = memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
    return memory_usage_by_variable
