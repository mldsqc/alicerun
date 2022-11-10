#!/usr/bin/env python3
# coding: utf-8

from datetime import datetime
from time import mktime
import sqlite3
import pylab as p
import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from scipy.signal import butter, filtfilt


class RawKindType:
    #0
    WALKING = 1
    CHARGING = 6
    NO_CHANGE_0 = 16
    WALKING_1 = 17 # happens only when walking… but what does it mean ?
    WALKING_2 = 18 # rare
    #26 # rare
    WAKEUP_0 = 28 # end of sleep, but stayed in the bed ?
    WALKING_3 = 33 # rare
    NO_STEPS = 80 # sitting, standing, or… sleeping
    #83 # looks to happens before 99, but not always
    #89
    #90 # XXX
    #91 # (80, 91, 112) means "Fall_asleep"
    WAKEUP_1 = 92
    NO_CHANGE_1 = 96
    NOT_WORN_0 = 99 # if appears at least 5 times, turns into 115
    #105 # rare
    #106 # happens days and nights
    NO_CHANGE_2 = 108
    LIGHT_SLEEP = 112
    NOT_WORN_1 = 115
    #121
    DEEP_SLEEP = 122
    FALL_ASLEEP = 123
    #124
    FALL_ASLEEP_PATTERNS = set(((NO_STEPS, 91, LIGHT_SLEEP),
                                (NO_STEPS, 91, NO_STEPS),
                               ))
    WAKEUP_PATTERNS = set(((LIGHT_SLEEP, NO_CHANGE_1, NO_STEPS),
                         ))

class ActivityType:
    UNKNOWN = -1
    NOT_WORN = 0
    WALKING = 10
    SITTING = 20
    SLEEPING = 30

def get_slices_of_consecutive_values(values):
    # """ >>> values = [1, 1, 2, 2, 2, 1, 3, 3]
    #     >>> res = list(get_slices_of_consecutive_values(values))
    #     >>> print(res)
    #     [slice(0, 2, None), slice(2, 5, None),
    #      slice(5, 6, None), slice(6, 8, None)]
    # """
    splits = p.append(p.where(p.diff(values) != 0)[0], len(values)+1)+1
    prev = 0
    for split in splits:
        yield slice(prev, split)
        prev = split

def filtered_serie(serie, freqmax):
    s = serie.interpolate()
    s = s[~s.isna()]

    dt = 60
    fs = 1/dt
    order = 5
    kind='low'
    nyq = 0.5*fs
    cuttoff = freqmax/nyq
    b, a = butter(order, cuttoff, btype=kind)

    filtered = filtfilt(b, a, s)
    return pd.Series(filtered, index=s.index)

def complete_heartrate(data):
    """ complete heartrate in data by interpolation where it is undefined """
    mask = ((data.HEART_RATE <= 0) | (data.HEART_RATE == 255))
    data.loc[mask, 'HEART_RATE'] = p.nan
    data.interpolate(inplace=True, kind='spline')

def complete_activity(data):
    """ add type, from raw_kind"""
    activities = p.zeros(len(data))
    fall_asleep_triggered = False

    NO_CHANGE = set((RawKindType.NO_CHANGE_0,
                     RawKindType.NO_CHANGE_1,
                     RawKindType.NO_CHANGE_2,
                   ))
    NOT_WORN = set((RawKindType.CHARGING,
                    RawKindType.NOT_WORN_0,
                    RawKindType.NOT_WORN_1,
                   ))
    SLEEPING = set((RawKindType.FALL_ASLEEP,
                    RawKindType.LIGHT_SLEEP,
                    RawKindType.DEEP_SLEEP,
                  ))
    SITTING = set((RawKindType.WAKEUP_0,
                   RawKindType.WAKEUP_1,
                ))
    WALKING = set((RawKindType.WALKING,
                   RawKindType.WALKING_1,
                   RawKindType.WALKING_2,
                 ))

    raw_kind = data.RAW_KIND
    for i, kind in enumerate(raw_kind):
        if i > 0 and kind in NO_CHANGE:
            activities[i] = activities[i-1]

            prev_current_next_kind = tuple(raw_kind.iloc[i-1:i+2])
            if prev_current_next_kind in RawKindType.WAKEUP_PATTERNS:
                activities[i] = ActivityType.SITTING

        elif kind in NOT_WORN:
            # if next type, is also NOT_WORN_0, then assuming not worn.
            # otherwise, just assume a mistake
            if kind == RawKindType.NOT_WORN_0:
                if raw_kind.iloc[i+1] in NOT_WORN:
                    activities[i] = ActivityType.NOT_WORN
                else:
                    activities[i] = activities[i-1]
            else:
                    activities[i] = ActivityType.NOT_WORN

        elif kind in SLEEPING:
            # check for a « fall_asleep » pattern
            three_prev_kind = tuple(raw_kind.iloc[i-2:i+1])
            if kind == RawKindType.FALL_ASLEEP or \
               three_prev_kind in RawKindType.FALL_ASLEEP_PATTERNS:
                fall_asleep_triggered = True

            if fall_asleep_triggered:
                activities[i] = ActivityType.SLEEPING
            else:
                activities[i] = ActivityType.SITTING

        elif kind in SITTING:
            activities[i] = ActivityType.SITTING

        elif kind in (RawKindType.NO_STEPS, ):
            if activities[i-1] == ActivityType.SLEEPING:
                activities[i] = ActivityType.SLEEPING
            else: # was walking, and stopped or was sitting and stayed sat.
                activities[i] = ActivityType.SITTING

            # check for a « fall_asleep » pattern
            three_prev_kind = tuple(raw_kind.iloc[i-2:i+1])
            if three_prev_kind in RawKindType.FALL_ASLEEP_PATTERNS:
                fall_asleep_triggered = True
                activities[i] = ActivityType.SLEEPING

        elif kind in WALKING:
            prev_current_next_kind = tuple(raw_kind.iloc[i-1:i+2])
            if data.STEPS.iloc[i] < 0:
                # sometimes, kind == WALKING, and steps == -1. It seems like a
                # bug in the record ; do not take it into account.
                activities[i] = activities[i-1]
            # this seems to be a pattern, saying « no change », like a bug…
            elif prev_current_next_kind == (RawKindType.NO_STEPS,
                                            RawKindType.WALKING,
                                            RawKindType.NO_STEPS):
                activities[i] = activities[i-1]
            else:
                activities[i] = ActivityType.WALKING
        else: # unknown RawKindType
            try:
                activities[i] = activities[i-1]
            except IndexError: #first activity, assuming SITTING
                activities[i] = ActivityType.SITTING

        if activities[i] != ActivityType.SLEEPING:
            fall_asleep_triggered = False

    previous_activity = activities[0]
    for s in get_slices_of_consecutive_values(activities):
        #remove 1 minute activities
        if len(activities[s]) <= 1:
            activities[s] = previous_activity

        if activities[s][0] == ActivityType.SLEEPING:
            if (data.RAW_INTENSITY.iloc[s] < 2).all():
                activities[s] = ActivityType.NOT_WORN

        previous_activity = activities[s][0]

    data['ACTIVITY'] = pd.Series(activities, index=data.index)


def complete_date(data):
    """ add the time and the date column, from timestamp """
    times =  []
    for index in data.index:
        times.append(index.time())

    data['TIME'] = pd.Series(times, index=data.index)
    data['DATETIME'] = pd.Series(data.index, index=data.index)

def remove_outliers(data):
    columns = ('STEPS', )
    eps = 1e-4
    for c in columns:
        qmax = data[c].quantile(1 - eps)
        qmin = data[c].quantile(eps)
        data = data[p.logical_and(qmin < data[c], data[c] < qmax)]
    return data

def loaddata(database, min_date=None, max_date=None, cached=True):
    """ Load miband data from the sqlite gadgetbrige database,
        if min_date is given, only events recorded after min_date will be
        loaded.
    """

    if cached and not min_date:
        try:
            data = pd.read_pickle(database + '.pkl')
            return data
        except FileNotFoundError:
            pass

    connection = sqlite3.connect(database)


    query = ('''select TIMESTAMP, RAW_INTENSITY, STEPS, RAW_KIND, HEART_RATE from MI_BAND_ACTIVITY_SAMPLE;'''
            )

    if min_date:
        try:
            min_timestamp = min_date.timestamp()
        except AttributeError:
            from dateparser import parse
            min_date = parse(min_date)
            min_timestamp = min_date.timestamp()
        query = query[:-1] + ' where TIMESTAMP >= {};'.format(min_timestamp)

    if max_date:
        try:
            max_timestamp = max_date.timestamp()
        except AttributeError:
            from dateparser import parse
            max_date = parse(max_date)
            max_timestamp = max_date.timestamp()
        if min_date:
            query = query[:-1] + ' and TIMESTAMP <= {};'.format(max_timestamp)
        else:
            query = query[:-1] + ' where TIMESTAMP <= {};'.format(max_timestamp)

    data = pd.read_sql(query, connection, index_col='TIMESTAMP', parse_dates=('TIMESTAMP'))
    data.index = data.index.tz_localize('UTC').tz_convert('Asia/Tbilisi')

    complete_heartrate(data)
    complete_activity(data)
    complete_date(data)
    #data = remove_outliers(data)

    mask_not_worn = data.ACTIVITY == ActivityType.NOT_WORN
    data.loc[mask_not_worn, 'HEART_RATE'] = p.nan
    data.loc[mask_not_worn, 'STEPS'] = p.nan
    data.loc[mask_not_worn, 'RAW_INTENSITY'] = p.nan

    if not min_date:
        data.to_pickle(database + '.pkl') # save the cache for future use
    return data

def get_shifts(data, activity_type):
    """ Iterate over shifts where the activity is `activity_type`
    """
    for s in get_slices_of_consecutive_values(data.ACTIVITY):
        current_activity_type = data.iloc[s].ACTIVITY[0]
        if current_activity_type == activity_type:
            yield data.iloc[s]
