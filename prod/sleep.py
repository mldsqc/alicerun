#!/usr/bin/env python3
# coding: utf-8

from datetime import timedelta

import pylab as p
import pandas as pd
from scipy.interpolate import interp1d

from model import loaddata, get_shifts, ActivityType

# Matsushima, Hiroyasu & Hirose, Kazuyuki & Hattori, Kiyohiko & Sato, Hiroyuki
# & Takadama, Keiki. (2012). Sleep Stage Estimation By Evolutionary Computation
# Using Heartbeat Data and Body-Movement. International Journal of Advancements
# in Computing Technology. 4. 281-290. 10.4156/ijact.vol4.issue22.31.

def requirement_1(sleep_shift):
    activity = sleep_shift.RAW_INTENSITY
    activity_std = activity.std()
    activity_std_rolling = activity.rolling(2, center=True).std()
    res = []
    cycle = False
    for rstd in activity_std_rolling:
        if rstd > activity_std:
            cycle = True
        elif rstd < 1:
            cycle = False
        res.append(cycle)
    return p.array(res)


def requirement_2(sleep_shift):
    timestamp = p.arange(len(sleep_shift))
    heartrate = sleep_shift.HEART_RATE

    A = p.vstack([timestamp, p.ones(len(timestamp))]).T
    m, c = p.lstsq(A, heartrate, rcond=None)[0] # heartrate = m * time + c

    return heartrate.rolling(10, center=True).mean() > m*timestamp + c

def get_sleep_score(sleep_df, sleep_duration_goal=8):
    """ return the sleep score of the given sleep_shift.

        The sleep score is high when :
            a) deepsleep + rem > 60% of the sleep_shift duration
            b) the sleep duration is equal to the `sleep_duration_goal`
            c) you did not move much during the night

        scores a, b and c are usually negative (except for the « perfect
        night »). The total score is defined by:
            total = 100 - score_a - score_b - score_c

        The sleep score is between 0 and 100
    """

    sleep_df_durations = sleep_df[['lightsleep', 'deepsleep', 'rem']]
    total_sleep = sleep_df_durations.sum(1)
    normalize_sleep_df = sleep_df_durations.div(total_sleep, 0)*100

    score_a = 60 - (normalize_sleep_df.deepsleep + normalize_sleep_df.rem)
    score_b = 100 * abs(total_sleep - sleep_duration_goal)/sleep_duration_goal
    score_c = sleep_df.raw_intensity_mean + sleep_df.raw_intensity_std/2

    score = pd.DataFrame(columns=('a', 'b', 'c', 'total'))


    weights = [1, 0.2, 2]

    score.a = -score_a * weights[0]
    score.b = -score_b * weights[1]
    score.c = -score_c * weights[2]
    score.total = 100 + score.a + score.b + score.c
    score.total[score.total < 0] = 0
    score.total[score.total > 100] = 100
    return score

def get_sleep_df(data, minduration=60):

    sleep_df = pd.DataFrame(columns=('lightsleep', 'deepsleep', 'rem',
                                     'raw_intensity_mean',
                                     'raw_intensity_std',
                                     'fall_asleep',
                                    ))
    indexes = []
    for i, sleep_shift in enumerate(get_shifts(data, ActivityType.SLEEPING)):

        if len(sleep_shift) < minduration:
            continue

        fall_asleep_datetime = sleep_shift.iloc[0].DATETIME
        fall_asleep_datetime -= timedelta(hours=12)
        indexes.append(fall_asleep_datetime.date())

        r1 = requirement_1(sleep_shift)
        r2 = requirement_2(sleep_shift)

        sleep_shift_start = sleep_shift.iloc[0].DATETIME
        sleep_shift_end = sleep_shift.iloc[-1].DATETIME

        sleep_state = r1 + 2*r2
        s0 = pd.Timedelta(minutes=sum(sleep_state == 0))
        s1 = pd.Timedelta(minutes=sum(sleep_state == 1))
        s2 = pd.Timedelta(minutes=sum(sleep_state == 2))
        s3 = pd.Timedelta(minutes=sum(sleep_state == 3))

        deepsleep = s0
        lightsleep = s1 + s2
        rem = s3

        sleep_info = [lightsleep.total_seconds()/3600,
                      deepsleep.total_seconds()/3600,
                      rem.total_seconds()/3600,
                      sleep_shift.RAW_INTENSITY.mean(),
                      sleep_shift.RAW_INTENSITY.std(),
                      sleep_shift.iloc[0].TIME,
                     ]
        sleep_df.loc[i] = sleep_info
    sleep_df.index = indexes
    return sleep_df


if __name__ == '__main__':
    data = loaddata('miband1.db') #, 'a week ago')
    sleep_df = get_sleep_df(data)

    sleep_df_durations = sleep_df[['lightsleep', 'deepsleep', 'rem']]

    sleep_df_durations.boxplot()
    sleep_df_durations.plot(kind='bar')

    normalize_sleep_df = sleep_df_durations.div(sleep_df_durations.sum(axis=1), axis=0)
    normalize_sleep_df.plot(kind='bar', stacked=True)

    p.show()
