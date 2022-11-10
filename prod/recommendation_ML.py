from data_preparing import *
from bot_answers_analysis import load_emotions_habits_values


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, SeasonalNaive, Naive,
    RandomWalkWithDrift, HistoricAverage )
import random


############### COLD start LOGIC

# 0.DONE detect by stats timewindows for best work,  and dinner. and walk outdoors
# 1.DONE detect area balance by week.
# 2.DONE detect witch areas are 'weak'
# 3.DONE sort tasks from each life area (make lists) from these areas by
#     - most resistance, most priority, most pleasure [MOST PLEASUREFUL]
#     - least resistance, least TTC, most pleasure  [EASY]
#     - most priority, most difficulty, most TTC  [MOST DIFFICULT]
# 4.DONE iterate by this lists.
# 5.DONE sometimes random
# 6.TODO recommend career tasks at the daytime
# 7.TODO make habits as recurring tasks, to include them to recommendation
# 8.TODO make morning routine
# 9.TODO make evening routine
# 10.TODO add tasks to calendar (maybe not all the day) up to next window end (input of emotions)
# make breaks  in tasks???
# 11.TODO sync recalculation of recommendation moments of emotion input
# 12.TODO add special very TASTY tasks (wishlist type, or bucketlist) to calendar somewhere in a week
#
#
# ############ MAYBE
# - Morning the most hated task (with most RES, but least by DIFF)
# - midday slow easy short
# - dinner fun social (or media) smth
# - afternoon mid - heavy mid short task
# - dinner fun or recreation or relation smth
# - midnight sex relax or intelligent.

def count_balance_of_life_areas_tasks_habits(df_tasks_3_t):
    """counting balance by life areas summing tasks and habits"""
    global df_emotions, df_habits, df_emotions1


    df_emotions, df_habits, df_emotions1 = emotions_habits_df_prepare()

    habits_values = load_emotions_habits_values()[1]
    result = {'PHYSICAL': 0, 'FUN_RECREATION': 0, 'INTELECTUAL': 0, 'LOVE ROMANCE SEX': 0, 'PARTNER': 0,
              'SOCIAL FRIENDS': 0,
              'FINANCIAL': 0, 'CAREER': 0}

    # last week counting tasks by life area name from todo tasks and registered habits
    for i in result.keys():
        # cheking if there are any completed tasks in this area
        if amount_complited_tasks_per_area_week(df_tasks_3_t, i).index[-1][0]:
            # print(i)
            number_area_tasks_weekly = amount_complited_tasks_per_area_week(df_tasks_3_t, i).iloc[-1, 0]
            # print(number_area_tasks_weekly)
            result[i] += number_area_tasks_weekly

    # list of habits done last week
    list_of_last_week_habits = df_habits.groupby(pd.Grouper(key='datetime_rnd', axis=0, freq='W'))\
                                                                        .agg({"act": lambda x:list(x)}).iloc[-1, 0]
    # print(list_of_last_week_habits)
    for k in list_of_last_week_habits:
        # print(k)
        # print(habits_values.values())
        for kk in habits_values.values():
            # print(kk.keys())
            if k in kk.keys():
                r = {i for i in habits_values if habits_values[i] == kk}
                r = ''.join(r)
                # print(r)
                result[r] += 1

    df_balance = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df_balance.columns.values[1] = 'amount'
    df_balance.sort_values(by=['amount'], inplace=True, ascending=True)

    # DF OF AREAS LESS THAN MEAN COMPARING WITH OTHER AREAS
    list_of_forgotten_areas = df_balance[df_balance.amount < df_balance.amount.mean()]['index']
    list_of_most_active_areas = df_balance[df_balance.amount > df_balance.amount.mean()]['index']

    return list_of_forgotten_areas, list_of_most_active_areas


# choose if metrics col is not empty
def return_tasks_list_by(df_tasks_3_t, life_area):
    """sort tasks from each life area (make lists) from these areas by
        - most resistance, most priority, most pleasure [MOST PLEASUREFUL]
        - least resistance, least TTC, most pleasure  [EASY]
        - most priority, most difficulty, most TTC  [MOST DIFFICULT]"""

    if df_tasks_3_t[(df_tasks_3_t.metricks.notna()) & (df_tasks_3_t.life_area == life_area)].shape[0] == 0 or\
            df_tasks_3_t[df_tasks_3_t['life_area'] == life_area].shape[0] == 0:

        return df_tasks_3_t[(df_tasks_3_t['life_area'] == life_area) & (
            df_tasks_3_t.status == 'NotStarted')] # .sample(n=3)# 3 random rows from list

    else:
        most_pleasureful_tasks_list = df_tasks_3_t[(df_tasks_3_t['life_area'] == life_area)
                                                   & (df_tasks_3_t['status'] =='NotStarted')].sort_values \
            (by=['RESIS', 'PRI', 'PLEAS'], ascending=[False, False, False]).subject.head(3)#.tolist()

        most_easy_tasks_list = df_tasks_3_t[(df_tasks_3_t['life_area'] == life_area)
                                            & (df_tasks_3_t['status'] =='NotStarted')].sort_values(
                                                                        by=['RESIS','TTC','PLEAS'],
                                                                        ascending=[True, True, False]).subject.head(3)
                                                                                                            #.tolist()

        most_difficult_tasks_list = df_tasks_3_t[(df_tasks_3_t['life_area'] == life_area)
                                                 & (df_tasks_3_t['status'] =='NotStarted')].sort_values(
                                                                        by=['PRI','DIFF','TTC'],
                                                                        ascending=[False, False, False]).subject.head(3)
                                                                                                            #.tolist()
        return most_pleasureful_tasks_list, most_easy_tasks_list, most_difficult_tasks_list


def cold_start_ml(df_tasks_3_t):

    most_forgotten = []
    most_active = []
    # recommend tasks from most forgotten areas
    for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[0]:
        if df_tasks_3_t[df_tasks_3_t['life_area'] == iii].shape[0] != 0:
            # print(iii)
            # print(' ------  ')
            # print(return_tasks_list_by(df_tasks_3_t, iii).subject.sample())
            most_forgotten.append('----TOP from ' + str(iii) + ':')
            # print('TOP from ' + str(iii) + ':' )
            # print(return_tasks_list_by(df_tasks_3_t, iii).subject.head(3))
            if type((return_tasks_list_by(df_tasks_3_t, iii))) == tuple:
                for kk in return_tasks_list_by(df_tasks_3_t, iii):
                    # print(k)
                    most_forgotten.extend(kk.to_list())
                    # most_forgotten += "\n"
                # print(most_forgotten)

            else:
                for ki in return_tasks_list_by(df_tasks_3_t, iii).subject.values:
                    # print(k)
                    most_forgotten.append(ki)
                    # most_forgotten += "\n"
                # print(most_forgotten)


    # recommend tasks from most active areas
    for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[1]:
        if df_tasks_3_t[df_tasks_3_t['life_area'] == iii].shape[0] != 0:

            # print(iii)
            # print(' ------  ')
            most_active.append('-----TOP from ' + str(iii) + ':')
            # most_active += "\n".join(return_tasks_list_by(df_tasks_3_t, iii))
            # most_active += "\n"
            # print(most_active)
            if len(return_tasks_list_by(df_tasks_3_t, iii))!=0:
                if type((return_tasks_list_by(df_tasks_3_t, iii))) == tuple:
                    for k in return_tasks_list_by(df_tasks_3_t, iii):
                        most_active.extend(k.to_list())
                        # print(most_active)
                        # print(k)
                # else:
                #     for ke in return_tasks_list_by(df_tasks_3_t, iii).subject.values:
                #         # print(k)
                #         most_forgotten.append(ke)

    return most_forgotten, most_active


# recommend most forgotten areas
# for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[0]:
#     print(return_tasks_list_by(df_tasks_3_t, iii))

# recommend least forgotten areas
# for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[1]:
#     print(iii)
#     print(return_tasks_list_by(df_tasks_3_t, iii))

# number_tasks_per_day_per_task(df_tasks_3_t)


def prepare_df_for_ml(df_tasks_3_t):
    """PREPARING THE DF TO TRAIN ON"""

    df_tasks_4_ml = df_tasks_3_t.copy()

    # dealing with not modified tasks modified column
    # df_tasks_4_ml[df_tasks_4_ml.body_last_modified.isnull()].body_last_modified = df_tasks_4_ml[df_tasks_4_ml
    #     .body_last_modified.isnull()].created_datetime

    df_tasks_4_ml.body_last_modified = df_tasks_4_ml.body_last_modified.fillna(df_tasks_4_ml.created_datetime)

    # datetime - modify to the day number,
    # last modified and completed difference in days
    # created and completed difference in days
    df_tasks_4_ml['created_datetime_numday'] = df_tasks_4_ml.created_datetime.dt.day_of_week
    df_tasks_4_ml['completed_datetime_numday'] = df_tasks_4_ml.completed_datetime.dt.day_of_week
    df_tasks_4_ml['body_last_modified_numday'] = df_tasks_4_ml.body_last_modified.astype('datetime64').dt.day_of_week
    df_tasks_4_ml['completed_minus_created_dayscount'] = (df_tasks_4_ml.completed_datetime - df_tasks_4_ml
                                                          .created_datetime).dt.days + 1
    df_tasks_4_ml['modified_minus_created_dayscount'] = (
                df_tasks_4_ml.body_last_modified.astype('datetime64') - df_tasks_4_ml.created_datetime).dt.days

    df_tasks_4_ml['status'] = df_tasks_4_ml['status'].astype('category')
    df_tasks_4_ml['importance'] = df_tasks_4_ml['importance'].astype('category')
    df_tasks_4_ml['group'] = df_tasks_4_ml['group'].astype('category')
    df_tasks_4_ml['life_area'] = df_tasks_4_ml['life_area'].astype('category')

    df_tasks_4_ml['body_last_modified'] = df_tasks_4_ml['body_last_modified'].astype('datetime64')

    df_tasks_4_ml['TTC'] = df_tasks_4_ml['TTC'].astype('Int32')
    df_tasks_4_ml['PRI'] = df_tasks_4_ml['PRI'].astype('Int32')
    df_tasks_4_ml['DIFF'] = df_tasks_4_ml['DIFF'].astype('Int32')
    df_tasks_4_ml['PLEAS'] = df_tasks_4_ml['PLEAS'].astype('Int32')
    df_tasks_4_ml['RESIS'] = df_tasks_4_ml['RESIS'].astype('Int32')
    df_tasks_4_ml['motivation'] = df_tasks_4_ml['motivation'].astype('Float32')

    df_tasks_4_ml['duration'] = df_tasks_4_ml['duration'] / 3600

    # get all categorical columns
    cat_columns = df_tasks_4_ml.select_dtypes(['category']).columns

    # convert all categorical columns to numeric
    df_tasks_4_ml[cat_columns] = df_tasks_4_ml[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # FILTERING OUT ROWS ON WICH IMPOSSIBLE TO TRAIN
    df_done_tasks_4_ml = df_tasks_4_ml[(df_tasks_4_ml.status == 1) & (df_tasks_4_ml.TTC.notna())]

    # NOT MUCH VALUES DEFINED BY FILTERING. EMOTIONS-DF STARTS FROM AUGUST,
    # SESSIONS AND TASKS DFS DIFFERS
    # 2022-07-05 - MORE TASKS , 2022-08-05 - START TIME TRACKING EMOTIONS
    df_done_tasks_5_ml = df_done_tasks_4_ml[df_done_tasks_4_ml.completed_datetime > datetime.datetime.strptime
    ('2022-07-05', '%Y-%m-%d')].copy()

    df_done_tasks_5_ml.drop(columns=['body_last_modified', 'created_datetime', 'completed_datetime', 'subject', 'metricks',
                                     'task_folder_local_id', 'original_body_content', 'duration'], inplace=True, axis=1)

    # df_done_tasks_5_ml
    # ,'motivation','duration','TTC_aft','PRI_aft','PLEASURE_aft','DIFCLT_aft', 'RES_aft'

    # CONVERTING_TYPES_FOR_CLASSIFICATION
    df_done_tasks_5_ml.TTC_aft.fillna(value=df_done_tasks_5_ml.TTC_aft.mean(), inplace=True)
    df_done_tasks_5_ml.TTC_aft = df_done_tasks_5_ml.TTC_aft.round(0).astype(int)
    df_done_tasks_5_ml.completed_datetime_numday = df_done_tasks_5_ml.completed_datetime_numday.round(0).astype(int)
    df_done_tasks_5_ml.completed_minus_created_dayscount = df_done_tasks_5_ml.completed_minus_created_dayscount.round(
        0).astype(int)
    df_done_tasks_5_ml.body_last_modified_numday = df_done_tasks_5_ml.body_last_modified_numday.round(0).astype(int)
    df_done_tasks_5_ml.modified_minus_created_dayscount = df_done_tasks_5_ml.modified_minus_created_dayscount.round(
        0).astype(int)
    df_done_tasks_5_ml.underrated = df_done_tasks_5_ml.underrated.astype(int)


    # MAKING PREDICTIONS ON NOT STARTED TASKS
    df_tasks_4_ml = df_tasks_4_ml[(df_tasks_4_ml.status == 0) & (df_tasks_4_ml.TTC.notna())]

    col_to_use_to_predict = ['importance', 'group', 'life_area', 'TTC', 'PRI', 'PLEAS', 'DIFF', 'RESIS',
                             'created_datetime_numday']

    df_tasks_4_ml.drop(columns=['body_last_modified', 'created_datetime', 'completed_datetime', 'subject', 'metricks',
                                'task_folder_local_id', 'original_body_content', 'duration'], inplace=True, axis=1)

    df_tasks_4_ml = df_tasks_4_ml.loc[:, col_to_use_to_predict]
    df_not_started_tasks_4_ml_prediction = df_tasks_4_ml[df_tasks_4_ml.PRI.notna()]

    return df_done_tasks_5_ml, df_not_started_tasks_4_ml_prediction


# KNN
def define_xy_for_knn(df_done_tasks_5_ml, target='TTC_aft'):
    col_to_use_to_predict = ['importance', 'group', 'life_area', 'TTC', 'PRI', 'PLEAS', 'DIFF', 'RESIS',
                             'created_datetime_numday']  # , 'body_last_modified_numday', 'modified_minus_created_dayscount'

    col_to_predict = ['motivation', 'TTC_aft', 'PRI_aft', 'PLEASURE_aft', 'DIFCLT_aft', 'RES_aft',
                      'underrated', 'completed_datetime_numday', 'completed_minus_created_dayscount']

    X,y = df_done_tasks_5_ml.loc[:, col_to_use_to_predict], df_done_tasks_5_ml.loc[:, target]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state = 2022)

    return X, y, X_train, X_test, y_train, y_test


def knn_clf(X_train, X_test, y_train, y_test, detect_best_k=1, plotting=0):
    """
        classification with KNN
        for each metric define better K
        plotting
        saving model in file

    """

    # detect_best_k for KNN
    if detect_best_k:
        # Model complexity
        neig = np.arange(1, 10)
        train_accuracy = []
        test_accuracy = []

        # Loop over different values of k
        for i, k in enumerate(neig):
            # k from 1 to 10(exclude)
            knn = KNeighborsClassifier(n_neighbors=k)
            # Fit with knn
            knn.fit(X_train,y_train)
            # train accuracy
            train_accuracy.append(knn.score(X_train, y_train))
            # test accuracy
            test_accuracy.append(knn.score(X_test, y_test))

        # Plot
        if plotting:
            plt.figure(figsize=[13,8])
            plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
            plt.plot(neig, train_accuracy, label = 'Training Accuracy')
            plt.legend()
            plt.title('-value VS Accuracy')
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            plt.xticks(neig)
            # plt.savefig('graph.png')
            plt.show()
            print(f"Best accuracy is {np.max(test_accuracy)} with K = {1+test_accuracy.index(np.max(test_accuracy))}")

        best_k = 1 + test_accuracy.index(np.max(test_accuracy))
        best_accuracy = np.max(test_accuracy)

    return best_k, best_accuracy


def save_load_predict_model_knn(X_train, X_test, y_train, y_test, best_k, best_accuracy,  df_to_predict,
                                target='TTC_aft',
                                train_model=1, save_model_file=1, load_model_and_predict=1):
    filename = target + '_' + str(best_k) + '_' + str(best_accuracy) + '.pkl'

    if train_model:
        knn_best = KNeighborsClassifier(n_neighbors=best_k)

        knn_best.fit(X_train, y_train)
        # prediction = knn_best.predict(df_to_predict)
        # print('Prediction: {}'.format(prediction))
        print('With KNN (K=) accuracy is: ', knn_best.score(X_test, y_test))  # accuracy

    #saving and using model
    if save_model_file:
        # Save the model as a pickle in a file
        joblib.dump(knn_best, filename)
        print('Saved model file as:', filename)

    if load_model_and_predict:
        # Load the model from the file
        knn_from_joblib = joblib.load('TTC_.pkl')

        # Use the loaded model to make predictions
        # modelscorev23.predict_proba(x_test)

        return knn_from_joblib.predict(df_to_predict)


# PREPARING DFs FOR KNN ON EMO-HABITS DATA
def prepare_data_tasks_sessions_emohabits(DB_EMOTIONS_TEST, df_tasks_3_t, df_sessions):
    # PREDICT PRODUCTIVE HOURS BASED ON 1-DONE TASKS and 2-NUMBER OF SESSIONS PER TIMESLOT

    def floor_dt(dt, interval=10):
        """rounding datetime column to 10 min intervals"""

        replace = (dt.minute // interval) * interval
        return dt.replace(minute=replace, second=0, microsecond=0)

    # DB_EMOTIONS_TEST = read_db_paths()[-1]
    emotions_habits = pd.read_csv(DB_EMOTIONS_TEST)
    emotions_values, habits_values, emotions_and_habits_values = load_emotions_habits_values()
    # emotions_habits

    emotions_habits['datetime'] = emotions_habits.date.astype('str') + ' ' + emotions_habits.time.astype('str')
    emotions_habits['datetime'] = emotions_habits['datetime'].astype('datetime64')

    emotions_habits.drop(['date', 'time'], axis=1, inplace=True)
    emotions_habits['datetime_rnd'] = emotions_habits.datetime.apply(floor_dt)
    emotions_habits['emo_hab'] = np.where(emotions_habits['act'].isin(emotions_values.keys()), 0, 1)

    emotions_habits = emotions_habits.groupby('datetime_rnd')['act'].agg(list).reset_index()  # .iloc[1,:]

    # GENERATION get dummies from list of emotions (act column) grouped by 10min periods
    emotions_habits = emotions_habits.join(emotions_habits['act'].str.join('|').str.get_dummies())

    emotions_habits.datetime_rnd = emotions_habits.datetime_rnd.dt.strftime('%Y-%m-%d %H:%M:%S')
    emotions_habits = emotions_habits.drop('act', axis=1)
    emotions_habits.set_index('datetime_rnd', inplace=True)
    # print(emotions_habits)

    def merge_close_timeslots(df):
        """MERGING THE MOST CLOSE REGISTERED ROWS-TIMESLOTS FOR EMOTIONS HABITS DF"""

        list_of_dd = df.index.to_list()
        # print(list_of_dd)

        list_of_id_todrop = []
        indd=0

        for kk in list_of_dd:
            if indd+2<=len(list_of_dd):
                deltaa=abs(pd.to_datetime(kk) - pd.to_datetime(list_of_dd[indd+1]))
                if deltaa < pd.Timedelta('1 hours'):
                    # print(df.loc[kk])
                    # print('---')
                    df.loc[kk] = df.loc[kk] + df.loc[list_of_dd[indd+1]]
                    list_of_id_todrop.append(list_of_dd[indd+1])

                    # print(df.loc[list_of_dd[indd+1]])
                    # print('---')
                    #
                    # print(df.loc[kk])
                    # print(indd, kk, list_of_dd[indd+1]) # for checking
                indd+=1
        for ki in list_of_id_todrop:
            df.drop(ki, inplace=True)
        return df

    emotions_habits = merge_close_timeslots(emotions_habits)

    # MAKING DF FROM SESSIONS
    def group_by_period(dff):
            dff['for_count']=dff['subject'].map(type) == str
            # number of sessions per day
            number_sessions_per_day = dff.groupby(pd.Grouper(key='start', axis=0,
                                                 freq='H')).agg({'for_count':sum}).reset_index()

            # amount of hours tracked per day
            amount_of_hours_tracked_per_day = dff.groupby([pd.Grouper(key='start', axis=0,
                                                 freq='H'),]).agg({'duration':sum})/ 3600
            # number_sessions_per_day.start = number_sessions_per_day.start.astype(str)

            out=pd.concat([number_sessions_per_day.set_index('start'),amount_of_hours_tracked_per_day], axis=1).reset_index()
            return out

    df_sessions_for_good = group_by_period(df_sessions)
    df_sessions_for_good = df_sessions_for_good.rename(columns={'for_count':'amount_sessions', 'start':'datetime_rnd'})
    df_sessions_for_good = df_sessions_for_good.drop(columns={'duration'})

    # FROM HERE WE ARE TAKING NUMBER OF SESSIONS IN SPECIFIC TIMESLOTS
    df_sessions_for_good.datetime_rnd = df_sessions_for_good.datetime_rnd.dt.strftime('%Y-%m-%d %H:%M:%S')

    df_sessions_for_good.set_index('datetime_rnd', inplace=True)

    # DETECTING NOT REGISTERED(in microsoft to-do) hours OF WHEN TASK WAS DONE
    # df_sessions1 = df_sessions.groupby(['subject']).agg({'stop': max}).reset_index()

    df_tasks_3_t_ = df_tasks_3_t.groupby(['completed_datetime']).agg({'subject':"value_counts"})
    # df_tasks_3_t_

    tasks_done_per_days = df_tasks_3_t_.groupby(['completed_datetime']).agg({'subject': sum}).reset_index()
    # tasks_done_per_days

    # ADDING HOURS MINUTES TO TASK_DONE DF BASED ON TIMESTAMPS IN EMOTIONS DF
    def add_random_hours_min(k):
        """detecting timeslots in emotions habits df to JOIN on them later"""
        list_of_timeslots = emotions_habits.reset_index().datetime_rnd.astype('datetime64').dt.strftime("%H:%M:%S").unique()
        return k +' '+ np.random.choice(list_of_timeslots)

    tasks_done_per_days.completed_datetime = tasks_done_per_days.completed_datetime.astype(str)\
                                                                            .apply(lambda x: add_random_hours_min(x))
    tasks_done_per_days.completed_datetime = tasks_done_per_days.completed_datetime.astype('datetime64')
    tasks_done_per_days = tasks_done_per_days.rename(columns={'completed_datetime': 'datetime_rnd',
                                                              'subject': 'tasks_done'})
    tasks_done_per_days = tasks_done_per_days.set_index('datetime_rnd')

    def find_nearest_date(df1, df_emo):
        """
        finding nearest date in df_emo and write it to df1

        #TODO too many times reseting the index. but now pipeline used in inner functions

        :param df1: df of sessions with generated hours minutes
        :param df_emo: emotions habits dataframe
        :return: df with changed time
        """

        df1 = df1.reset_index()
        df_emo = df_emo.reset_index()
        df1.datetime_rnd = df1.datetime_rnd.astype('datetime64')
        df_emo.datetime_rnd = df_emo.datetime_rnd.astype('datetime64')

        df2 = df1.copy() # CONTroVERSIVE

        # return df1.datetime_rnd
        for i in df2.datetime_rnd:
            # print(i)
            minidx_ = abs(i - df_emo['datetime_rnd']).argmin()
            delta = abs(i - df_emo.loc[minidx_,'datetime_rnd'])
            # print(i, df_emo.loc[[minidx_]]['datetime_rnd'] , delta, delta/pd.Timedelta('1 hour'))

            if delta/pd.Timedelta('1 hour') < 5:
                index = df2.index[df2.datetime_rnd==i]
                # print(df1.loc[index, 'datetime_rnd'])
                # print('BEFORE',df_emo.loc[minidx_,'datetime_rnd'])
                df2.loc[index, 'datetime_rnd'] = df_emo.loc[minidx_,'datetime_rnd']
                # print('CHANGE', df1.loc[index, 'datetime_rnd'])

            # print()
            # print(df_emo.loc[[minidx_]])
        return df2

    df_sessions_for_good_ = find_nearest_date(df_sessions_for_good, emotions_habits)
    tasks_done_per_days_ = find_nearest_date(tasks_done_per_days, emotions_habits)

    # summing up duplicates
    df_sessions_for_good_ = df_sessions_for_good_.reset_index().groupby('datetime_rnd').agg({'amount_sessions': sum})

    # just test
    # df_sessions_for_good_[df_sessions_for_good_.datetime_rnd=='2022-08-09 10:50:00']
    # df_sessions_for_good_[df_sessions_for_good_.amount_sessions>0]

    # df_sessions_for_good_ = df_sessions_for_good_.set_index('datetime_rnd')
    # tasks_done_per_days_ = tasks_done_per_days_.set_index('datetime_rnd')

    emotions_habits_t = emotions_habits.reset_index().copy()
    emotions_habits_t.datetime_rnd = emotions_habits_t.datetime_rnd.astype(str)

    tasks_done_per_days_.datetime_rnd = tasks_done_per_days_.datetime_rnd.astype(str)
    # emotions_habits_t

    df_sessions_for_good_t = df_sessions_for_good_.reset_index().copy()
    df_sessions_for_good_t.datetime_rnd = df_sessions_for_good_t.datetime_rnd.astype(str)

    # MERGING ALL together
    df_final_ = emotions_habits_t.merge(tasks_done_per_days_, how='left', on='datetime_rnd')
    df_final = df_final_.merge(df_sessions_for_good_t, how='left', on='datetime_rnd')

    return df_final


def prepare_emohab_2_for_ML_knn(df_final):
    """Adding moon phases"""

    # dropping 'tasks_done' because too small amount
    col_to_use2 = ['+fear', '+sadness', '+surprise', '+youtubed', '-fear',
                   '-sadness', '-surprise', '10 pushups everyday',
                   '5 minute journal', 'Full-time', 'Part-time',
                   'UNPREDICTABLE EMOTIONAL WOWs', 'admiration', 'amusement', 'anger',
                   'anxiety', 'any pain', 'appreciation', 'arguing',
                   'big physical activity', 'boredom', 'burnouted', 'cafe', 'calmness',
                   'chess', 'cold shower', 'common goals completion', 'confusion',
                   'critiqued_by_HER', 'critiqued_by_ME', 'desire', 'disgust',
                   'dissatisfaction', 'drugs', 'embarrassed', 'emotionally UNbalanced',
                   'emotionally balanced', 'empathic', 'excitement', 'fascination',
                   'fastfood', 'film', 'financial reduce costs', 'focused', 'happiness',
                   'inspiration', 'interest', 'joy', 'jrk', 'languages',
                   'look inside 4 feelings on whole life', 'made smth for selfefficiency',
                   'meditation', 'motivated', 'new people', 'new sex partner', 'nostalgia',
                   'old_friends', 'opensourced questions answered', 'over_eated', 'pain',
                   'pleasure from done tasks', 'porn', 'pride', 'procrastinated',
                   'productive', 'psycho practices', 'reading', 'relief', 'romance',
                   'running on plans feeling', 'satisfaction', 'sex', 'sexual desire',
                   'slept_GOOD', 'social offline', 'studing', 'surprise', 'too much news',
                   'too much social media', 'too much youtube', 'traveled', 'tvshow',
                   'work thru complicated situations', 'Не можу працювати',
                   'day', 'hour', 'moon_phase']

    col_to_predict2_positive = ['amount_sessions', '+surprise', 'amusement', 'drugs',
                                'emotionally balanced', 'focused', 'happiness', 'motivated',
                                'new people', 'pleasure from done tasks', 'pride',
                                'productive', 'running on plans feeling', ]

    col_to_predict2_negative = ['burnouted', 'Не можу працювати', 'anxiety', 'any pain',
                                'confusion', 'emotionally UNbalanced', 'procrastinated', ]

    cols_to_drop2 = ['datetime_rnd', '0-2', '5-7', '8-10', '2-5', 'tasks_done', ]

    def moon_phase(datetime_=None):
        """
        https://gist.github.com/miklb/ed145757971096565723
        :return: 1 out of 8  moon phases
        """
        import math, decimal, datetime
        dec = decimal.Decimal

        def position(now=None):
            if now is None:
                now = datetime.datetime.now()

            diff = now - datetime.datetime(2001, 1, 1)
            days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
            lunations = dec("0.20439731") + (days * dec("0.03386319269"))

            return lunations % dec(1)

        def phase(pos):
            index = (pos * dec(8)) + dec("0.5")
            index = math.floor(index)
            return index
            # return {
            #    0: "New Moon",
            #    1: "Waxing Crescent",
            #    2: "First Quarter",
            #    3: "Waxing Gibbous",
            #    4: "Full Moon",
            #    5: "Waning Gibbous",
            #    6: "Last Quarter",
            #    7: "Waning Crescent"
            # }[int(index) & 7]

        pos = position(datetime_)
        phasename = phase(pos)

        roundedpos = round(float(pos), 3)
        # print ("%s (%s)" % (phasename, roundedpos))
        return phasename

    # preparing df for ML
    df_hab_emo_for_ML = df_final.copy()

    df_hab_emo_for_ML['day'] = df_hab_emo_for_ML.datetime_rnd.astype('datetime64').dt.day_of_week
    df_hab_emo_for_ML['hour'] = df_hab_emo_for_ML.datetime_rnd.astype('datetime64').dt.hour

    # making moonphases
    df_hab_emo_for_ML['moon_phase'] = df_hab_emo_for_ML.datetime_rnd.apply(lambda x: moon_phase(pd.to_datetime(x)))

    # merging to 1 categorical column
    df_hab_emo_for_ML['common_energy'] = df_hab_emo_for_ML['0-2'] + 2 * df_hab_emo_for_ML['2-5'] \
                                         + 3 * df_hab_emo_for_ML['5-7'] + 4 * df_hab_emo_for_ML['8-10']

    # dropping cols
    for kil in cols_to_drop2:
        df_hab_emo_for_ML.drop(kil, axis=1, inplace=True)

    # converting to categorical
    for convo in col_to_use2:
        df_hab_emo_for_ML[convo] = df_hab_emo_for_ML[convo].astype('category')
    return df_hab_emo_for_ML


def predict_timeseries_emohab(df_final):
    """graphs timeseries for emotions habits data"""
    def plot_grid(df_train, df_test=None, plot_random=True, model=None, level=None):
        from itertools import product

        fig, axes = plt.subplots(4, 2, figsize=(24, 14))

        unique_ids = df_train['unique_id'].unique()

        assert len(unique_ids) >= 8, "Must provide at least 8 ts"

        if plot_random:
            unique_ids = random.sample(list(unique_ids), k=8)
        else:
            unique_ids = unique_ids[:8]

        for uid, (idx, idy) in zip(unique_ids, product(range(4), range(2))):
            train_uid = df_train.query('unique_id == @uid')
            axes[idx, idy].plot(train_uid['ds'], train_uid['y'], label='y_train')
            if df_test is not None:
                max_ds = train_uid['ds'].max()
                test_uid = df_test.query('unique_id == @uid')
                for col in ['y', model, 'y_test']:
                    if col in test_uid:
                        axes[idx, idy].plot(test_uid['ds'], test_uid[col], label=col)
                if level is not None:
                    for l, alpha in zip(sorted(level), [0.5, .4, .35, .2]):
                        axes[idx, idy].fill_between(
                            test_uid['ds'],
                            test_uid[f'{model}-lo-{l}'],
                            test_uid[f'{model}-hi-{l}'],
                            alpha=alpha,
                            color='orange',
                            label=f'{model}_level_{l}',
                        )
            axes[idx, idy].set_title(f'M4 Hourly: {uid}')
            axes[idx, idy].set_xlabel('Timestamp [t]')
            axes[idx, idy].set_ylabel('Target')
            axes[idx, idy].legend(loc='upper left')
            axes[idx, idy].xaxis.set_major_locator(plt.MaxNLocator(20))
            axes[idx, idy].grid()
        fig.subplots_adjust(hspace=0.5)
        plt.show()

    train_days_on = '2 days'

    df_for_timeseries = df_final.copy().set_index('datetime_rnd')

    # filtering and dropping too small data columns
    drop_level = 10

    to_drop = []
    for i in df_for_timeseries.columns:
        if df_for_timeseries[i].value_counts()[1] < drop_level:
            to_drop.append(i)

    for kl in to_drop:
        df_for_timeseries.drop(columns={kl}, inplace=True)

    # how many features left after dropping
    number_of_series = df_for_timeseries.shape[1]

    df_for_timeseries = df_for_timeseries.stack().reset_index()

    df_for_timeseries = df_for_timeseries.rename(columns={'level_1': 'unique_id', 0: 'y', 'datetime_rnd': 'ds'})

    # onehotencoding
    # df_for_timeseries.unique_id = df_for_timeseries.unique_id.astype('category').cat.codes#pd.factorize(df_for_timeseries.unique_id)[0]
    # df_for_timeseries = df_for_timeseries.set_index('unique_id')

    df_for_timeseries.ds = df_for_timeseries.ds.astype('datetime64')

    # how  many days to take for train set
    dayx = pd.to_datetime(df_for_timeseries.ds.max().strftime('%Y-%m-%d')) - pd.Timedelta(train_days_on)

    Y_train_df = df_for_timeseries[df_for_timeseries.ds < dayx]
    Y_test_df = df_for_timeseries[df_for_timeseries.ds > dayx]

    n_series = number_of_series
    uids = Y_train_df['unique_id'].unique()[:n_series]
    train = Y_train_df.query('unique_id in @uids')
    test = Y_test_df.query('unique_id in @uids')
    # train

    # plot_grid(train, test)

    models = [
        AutoARIMA(season_length=24, approximation=True),
        Naive(),
        SeasonalNaive(season_length=24),
        RandomWalkWithDrift(),
        HistoricAverage()
    ]

    fcst = StatsForecast(df=train,
                         models=models,
                         freq='H',
                         n_jobs=-1)

    # setting levels of probability
    levels = [95, 99]
    forecasts = fcst.forecast(h=48, level=levels)
    forecasts = forecasts.reset_index()
    # forecasts.head()

    test = test.merge(forecasts, how='left', on=['unique_id', 'ds'])

    plot_grid(train, test, level=levels, model='AutoARIMA')
    plot_grid(train, test, level=levels, model='SeasonalNaive')
    plot_grid(train, test, level=levels, model='HistoricAverage')
    plot_grid(train, test, level=levels, model='Naive')
    plot_grid(train, test, level=levels, model='RWD')


def correlation_matrix_for_emohabits(df_final):
    corr = df_final.corr()
    corr.style.background_gradient(cmap='coolwarm')


def print_highly_correlated(df, features, threshold=0.3):
    """Prints highly correlated features pairs in the data frame (helpful for feature engineering)
        pairs of highly correlated features
     Usage:
     e.g. print_highly_correlated(df=model, features=model.columns)"""

    corr_df = df[features].corr()  # get correlations
    correlated_features = np.where(np.abs(corr_df) > threshold) # select ones above the abs threshold
    correlated_features = [(corr_df.iloc[x,y], x, y) for x, y in zip(*correlated_features) if x != y and x < y] # avoid duplication
    s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))  # sort by correlation value

    list_of_correlations = []

    if not s_corr_list:
        print("There are no highly correlated features with correlation above", threshold)
    else:
        for vv, ii, jj in s_corr_list:
            cols = df[features].columns
            # print("%s and %s = %.3f" % (corr_df.index[ii], corr_df.columns[jj], vv))
            list_of_correlations.append("%s and %s = %.3f" % (corr_df.index[ii], corr_df.columns[jj], vv))

    return random.choice(list_of_correlations[1:])


def knn_predict_sessions_amount_based_on_emotions(df_final):
    """or how emotions influence(?) on amount of work sessions?
    KNN prediction for 'amount_sessions'
    """

    # dropping 'tasks_done' because too small amount
    col_to_use2 = ['+fear', '+sadness', '+surprise', '+youtubed', '-fear',
                   '-sadness', '-surprise', '10 pushups everyday',
                   '5 minute journal', 'Full-time', 'Part-time',
                   'UNPREDICTABLE EMOTIONAL WOWs', 'admiration', 'amusement', 'anger',
                   'anxiety', 'any pain', 'appreciation', 'arguing',
                   'big physical activity', 'boredom', 'burnouted', 'cafe', 'calmness',
                   'chess', 'cold shower', 'common goals completion', 'confusion',
                   'critiqued_by_HER', 'critiqued_by_ME', 'desire', 'disgust',
                   'dissatisfaction', 'drugs', 'embarrassed', 'emotionally UNbalanced',
                   'emotionally balanced', 'empathic', 'excitement', 'fascination',
                   'fastfood', 'film', 'financial reduce costs', 'focused', 'happiness',
                   'inspiration', 'interest', 'joy', 'jrk', 'languages',
                   'look inside 4 feelings on whole life', 'made smth for selfefficiency',
                   'meditation', 'motivated', 'new people', 'new sex partner', 'nostalgia',
                   'old_friends', 'opensourced questions answered', 'over_eated', 'pain',
                   'pleasure from done tasks', 'porn', 'pride', 'procrastinated',
                   'productive', 'psycho practices', 'reading', 'relief', 'romance',
                   'running on plans feeling', 'satisfaction', 'sex', 'sexual desire',
                   'slept_GOOD', 'social offline', 'studing', 'surprise', 'too much news',
                   'too much social media', 'too much youtube', 'traveled', 'tvshow',
                   'work thru complicated situations', 'Не можу працювати',
                   'day', 'hour', 'moon_phase']

    col_to_predict2_positive = ['amount_sessions', '+surprise', 'amusement', 'drugs',
                                'emotionally balanced', 'focused', 'happiness', 'motivated',
                                'new people', 'pleasure from done tasks', 'pride',
                                'productive', 'running on plans feeling', ]

    col_to_predict2_negative = ['burnouted', 'Не можу працювати', 'anxiety', 'any pain',
                                'confusion', 'emotionally UNbalanced', 'procrastinated', ]

    cols_to_drop2 = ['datetime_rnd', '0-2', '5-7', '8-10', '2-5', 'tasks_done', ]

    df_hab_emo_for_ML = prepare_emohab_2_for_ML_knn(df_final)

    X, y = df_hab_emo_for_ML.loc[:, col_to_use2], df_hab_emo_for_ML.loc[:, 'amount_sessions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022)

    best_kk = knn_clf(X_train, X_test, y_train, y_test, detect_best_k=1)[0]
    best_accuracy_ = knn_clf(X_train, X_test, y_train, y_test, detect_best_k=1)[1]

    filename = 'amount_sessions' + '_' + str(best_kk) + '_' + str(best_accuracy_) + '.pkl'
    print('Saved model file as:', filename)

    knn_best = KNeighborsClassifier(n_neighbors=best_kk)
    knn_best.fit(X_train, y_train)
    print('With KNN (K=) accuracy is: ', knn_best.score(X_test, y_test))  # accuracy

    # saving and using model
    # Save the model as a pickle in a file
    joblib.dump(knn_best, filename)

    # Load the model from the file
    knn_from_joblib = joblib.load(filename)

    # Use the loaded model to make predictions
    # modelscorev23.predict_proba(x_test)
    knn_from_joblib.predict(X_test)

    # TODO fix later
    # get predictions how many work sessions i will make, based on habits emotions
    # df_not_started_tasks_4_ml_prediction['TTC_pred'] = save_load_predict_model_knn(
    #     best_kk, best_accuracy_,
    #     target='amount_sessions',
    #     df_to_predict=df_hab_emo_for_ML,
    #     train_model=0, save_model_file=0,
    #     load_model_and_predict=1)
    #
    # # sorting and getting index of tasks to get task names back
    # df_not_started_tasks_4_ml_prediction.sort_values(by=['amount_sessions'], inplace=True, ascending=False)  # .index
    # indexes = df_not_started_tasks_4_ml_prediction.index
    # # indexes
    # list_knn_tasks = df_tasks_3.iloc[indexes, 0].to_list()
