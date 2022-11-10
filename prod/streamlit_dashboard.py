
# import pylab as p
# import matplotlib
from matplotlib import pylab as p
from pylab import *

from datetime import timedelta

# GRAPH AND DASHBOARD LIBS
import plotly.tools as tls
import plotly.graph_objects as go
from plotly.colors import n_colors

# import dash
# import dash_bootstrap_components as dbc
# from dash_bootstrap_templates import load_figure_template
# from dash import dcc
# from dash import html

import streamlit as st
import streamlit.components.v1 as components

# from collections import defaultdict as _defaultdict
# from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
# from typing import Dict as _Dict
# import streamlit as _st
# import sys as _sys

# OTHER FLOWS
from recommendation_ML import *
from gant_diagram import *
# from gcal import *

from data_preparing import *
# from bot_answers_analysis import load_emotions_habits_values
from model import loaddata, get_shifts, ActivityType, filtered_serie
from sleep import get_sleep_df, get_sleep_score

import re
import warnings

warnings.filterwarnings('ignore')

# TODO write right path in DBs folder DB_GADGETBRIDGE
data = loaddata('./android_db/miband.db', cached=True)

fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()
fig4 = go.Figure()
fig5 = go.Figure()
fig6 = go.Figure()
fig7 = go.Figure()


def random_emoji():
    """# callback to update emojis in Session State
    # in response to the on_click event"""
    emojis = ["💯", "✨", "❗", "🆗", "🆘", "🆙", "🌄", "🌅", "🌈", "🌋", "🌌", "🌊", "🌞", "🌩️",
              "🌱", "🌶️", "🍩", "🍰", "🎇", "🎚️", "🎛️", "🎲", "🌩️"]
    return random.choice(emojis)


##### PLOT FUNCTIONS FOR SMARTWATCH DATA ##########
# @task(retries=2, retry_delay_seconds=5)
def _plot_mean(data, attribute, name, window, fig_name):
    fig = fig_name

    mean = getattr(data, attribute).interpolate().rolling(window).mean()
    std = getattr(data, attribute).interpolate().rolling(window).std()
    filtrd = filtered_serie(getattr(data, attribute), 5e-6)

    fig.add_trace(go.Scatter(x=data.index, y=getattr(data, attribute)))  # , line_shape=dict({'spline'})
    fig.add_trace(go.Scatter(x=mean.index, y=mean, mode='lines', line=dict(color='rgb(31, 119, 180)')))
    fig.add_trace(go.Scatter(x=mean.index, y=mean + std, fill='tonexty',
                             fillcolor='rgba(0,100,80,0.4)')).add_trace(go.Scatter(x=mean.index, y=mean - std,
                                                                                   fill='tonexty',
                                                                                   fillcolor='rgba(0,100,80,0.4)'))

    fig.add_trace(go.Scatter(x=filtrd.index, y=filtrd.values))

    fig.update_layout(
        title=name, xaxis_title="", yaxis_title="",
        height=450, width=250, margin=dict(t=55, l=15, b=15, r=15),
        showlegend=False
    ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    return fig


# @task(retries=2, retry_delay_seconds=5)
def _plot_boxplot_by_weekday(data, attribute, fig_name, function='sum', plot_name=''):
    # pd.options.plotting.backend = "plotly"

    fig = fig_name
    by_day = getattr(data.groupby(data.index.date), attribute)
    results_by_day = getattr(by_day, function)()
    results_by_day = pd.DataFrame(results_by_day, index=pd.to_datetime(results_by_day.index))
    results_by_day['weekday'] = results_by_day.apply(lambda x: x.index.weekday)
    # print( results_by_day.head(10))
    # results_by_day.boxplot(by='weekday').show()

    # fig.add_trace(go.Box(y=results_by_day.weekday))
    for i, day in enumerate(results_by_day['weekday'].unique()):
        df_plot = results_by_day[results_by_day['weekday'] == day]
        # print(df_plot.head())

        fig.add_trace(go.Box(x=df_plot['weekday'], y=df_plot[attribute]))
    # fig.add_trace(go.Box(y=results_by_day[attribute]))
    # fig = px.box(results_by_day, x='weekday')

    fig.update_layout(
        title=plot_name, xaxis_title="Date", yaxis_title="", boxmode='group', xaxis_tickangle=0,
        height=400, width=250,margin=dict(t=55,l=15,b=15,r=15), showlegend=False
    ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_mean_heart_rate(data, window=60 * 24):
    _plot_mean(data, 'HEART_RATE', 'HR', window, fig_name=fig1)
    # p.legend()


# @task(retries=2, retry_delay_seconds=5)
def plot_mean_raw_intensity(data, window=60 * 24):
    _plot_mean(data, 'RAW_INTENSITY', 'Average raw intensity', window, fig_name=fig2)
    # p.legend()


# @task(retries=2, retry_delay_seconds=5)
def plot_steps_boxplot_by_weekday(data):
    _plot_boxplot_by_weekday(data, 'STEPS', fig3, 'sum', 'Summed steps by weekday')


# @task(retries=2, retry_delay_seconds=5)
def plot_heart_rate_boxplot_by_weekday(data):
    _plot_boxplot_by_weekday(data, 'HEART_RATE', fig4, 'mean', 'Mean HEART RATE by weekday')


# @task(retries=2, retry_delay_seconds=5)
def plot_activity_boxplot_by_weekday(data):
    _plot_boxplot_by_weekday(data, 'RAW_INTENSITY', fig5, 'mean', 'Mean activity by weekday')


# @task(retries=2, retry_delay_seconds=5)
def plot_mean_week(data):
    global fig6
    pd.plotting.register_matplotlib_converters()
    fig, axis = p.subplots(3, 3, sharex=True, sharey=True)

    for (day, data_day) in data.groupby(data.index.weekday):
        ax = axis[p.unravel_index(day, axis.shape)]

        data_day_by_time = data_day.groupby('TIME')

        steps = data_day_by_time.STEPS.mean()
        intensity = data_day_by_time.RAW_INTENSITY.mean().rolling(20).mean()
        hr = data_day_by_time.HEART_RATE.mean().rolling(20).mean()
        activity = data_day_by_time.ACTIVITY.mean().rolling(5).mean()

        ax.plot(activity, c='C1', alpha=0.2)
        ax.plot(steps, c='C0', alpha=0.9)
        ax.plot(intensity, c='C2', alpha=0.7)
        ax.plot(hr, c='C3')

    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['showlegend'] = True

    fig6 = go.Figure(plotly_fig)
    fig6.update_layout(
        title="Last Week activity, Hear rate", xaxis_title="", yaxis_title="", boxmode='group',
        xaxis_tickangle=0, showlegend=False
    ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    return fig6


# @task(retries=2, retry_delay_seconds=5)
def plot_cumstep_by_day(data, goal=8000):
    pd.options.plotting.backend = "plotly"

    gd = data.groupby(data.index.date)
    fig7_ = data.RAW_INTENSITY.plot(title='Intensity of actions',
                                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                                    template="plotly_dark").update_layout(showlegend=False, xaxis_title='',
                                                                          yaxis_title='', height=300,
                                                                          width=250, margin=dict(t=55, l=15, b=15, r=15)
                                                                          ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})

    fig7__ = gd.STEPS.cumsum().plot(title='Amount of steps',
                                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                                    template="plotly_dark")\
        .update_layout(showlegend=False, yaxis_title='', height=300, width=250, margin=dict(t=55, l=15, b=15, r=15)
                        ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})

    return [fig7__, fig7_]


# @task(retries=2, retry_delay_seconds=5)
def plot_calendar_of_steps(data, goal=8000):
    calendar = p.ones((52, 7)) * p.nan

    steps_by_day = data.groupby(data.index.date).STEPS.sum()

    min_week = 52
    max_week = 0
    for day, steps in steps_by_day.items():
        weeknumber = day.isocalendar()[1]
        weekday = day.isoweekday() - 1

        min_week = min(weeknumber, min_week)
        max_week = max(weeknumber, max_week)
        calendar[weeknumber, weekday] = steps

    p.title('Number of steps per day')
    vmax = max(2 * goal, p.nanmax(calendar))
    vmin = p.nanmin(calendar)
    # p.colorbar(orientation='horizontal')
    # p.imshow(calendar[min_week:max_week+1, :], vmin=vmin, vmax=vmax,
    #         aspect='auto',
    #         extent=(0, 6, max_week, min_week))
    fig = px.imshow(calendar[min_week:max_week + 1, :], zmin=vmin, zmax=vmax,
                    aspect='auto', title='Number of steps per day', template="plotly_dark",)\
        .update_layout(showlegend=False, yaxis_title='')
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_calendar_of_activity(data, goal=8000):
    calendar = p.ones((52, 7)) * p.nan

    activity_by_day = data.groupby(data.index.date).RAW_INTENSITY.sum()

    min_week = 52
    max_week = 0
    for day, activity in activity_by_day.items():
        weeknumber = day.isocalendar()[1]
        weekday = day.isoweekday() - 1

        min_week = min(weeknumber, min_week)
        max_week = max(weeknumber, max_week)
        calendar[weeknumber, weekday] = activity

    p.title('Number of activity per day')
    vmax = max(2 * goal, p.nanmax(calendar))
    vmin = p.nanmin(calendar)

    p.imshow(calendar[min_week:max_week+1, :], vmin=vmin, vmax=vmax,
             aspect='auto', extent=(0, 6, max_week, min_week))
    p.colorbar(orientation='horizontal')

    fig = px.imshow(calendar[min_week:max_week + 1, :], zmin=vmin, zmax=vmax, aspect='auto',
                    title='Number of activity per day', template="plotly_dark",)

    # fig.show()
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_calendar_of_sleep_duration(data, goal=9.5):
    calendar = p.nan * p.ones((52, 7))
    min_week = 52
    max_week = 0

    for sleep in get_shifts(data, ActivityType.SLEEPING):
        fall_asleep_datetime = sleep.iloc[0].DATETIME
        wake_up_datetime = sleep.iloc[-1].DATETIME
        duration = (wake_up_datetime - fall_asleep_datetime).total_seconds() / (60 * 60)

        # Substract 12h, to be sure we got the good date. Eg:
        # Fall asleep | Fall asleep - 12h | date to be used
        # 04/23 23:20 |    04/23 11:30    |      04/23
        # 04/25 03:20 |    04/24 15:20    |      04/24
        # 04/25 22:00 |    04/25 22:20    |      04/25

        fall_asleep_datetime -= timedelta(hours=12)
        weeknumber = fall_asleep_datetime.isocalendar()[1]
        weekday = fall_asleep_datetime.isoweekday() - 1

        min_week = min(weeknumber, min_week)
        max_week = max(weeknumber, max_week)
        if p.isnan(calendar[weeknumber, weekday]):
            calendar[weeknumber, weekday] = duration
        else:
            calendar[weeknumber, weekday] += duration

    p.title('sleep duration')
    vmax = min(p.nanmax(calendar), 12)
    vmin = p.nanmin(calendar)
    # p.imshow(calendar[min_week:max_week+1, :], vmin=vmin, vmax=vmax,
    #         aspect='auto', extent=(0, 6, max_week, min_week))
    # p.colorbar(orientation='horizontal')
    # p.show()
    fig = px.imshow(calendar[min_week:max_week + 1, :], zmin=vmin, zmax=vmax,
                    aspect='auto', title='Sleep Duration', template="plotly_dark",)
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_calendar_of_deepsleep_percentage(data, goal=7.5):
    calendar = p.nan * p.ones((52, 7))
    min_week = 52
    max_week = 0

    for date, durations in get_sleep_df(data).iterrows():
        weeknumber = date.isocalendar()[1]
        weekday = date.isoweekday() - 1

        total = durations.lightsleep + durations.deepsleep + durations.rem
        deepsleep_percent = durations.rem / total * 100

        calendar[weeknumber, weekday] = deepsleep_percent

        min_week = min(weeknumber, min_week)
        max_week = max(weeknumber, max_week)

    p.title('deepsleep duration (%)')
    vmax = None
    vmin = None
    # p.imshow(calendar[min_week:max_week+1, :], vmin=vmin, vmax=vmax,
    #          aspect='auto', extent=(0, 6, max_week, min_week))
    # p.colorbar(orientation='horizontal')

    fig = px.imshow(calendar[min_week:max_week + 1, :], zmin=vmin, zmax=vmax,
                    aspect='auto', title='Deepsleep Duration (%)', template="plotly_dark",)
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_calendar_of_sleep_score(data, goal=7.5):
    calendar = p.nan * p.ones((52, 7))
    min_week = 52
    max_week = 0

    for date, sc in get_sleep_score(get_sleep_df(data), goal).iterrows():
        weeknumber = date.isocalendar()[1]
        weekday = date.isoweekday() - 1
        calendar[weeknumber, weekday] = sc.total

        min_week = min(weeknumber, min_week)
        max_week = max(weeknumber, max_week)

    p.title('sleep score')
    vmax = 100
    vmin = min(60, p.nanmin(calendar))
    # p.imshow(calendar[min_week:max_week+1, :], vmin=vmin, vmax=vmax,
    #          cmap='cool', aspect='auto', extent=(0, 6, max_week, min_week))
    # p.colorbar(orientation='horizontal')
    fig = px.imshow(calendar[min_week:max_week + 1, :], zmin=vmin, zmax=vmax,
                    aspect='auto', title='Sleep Score', template="plotly_dark",)
    return fig


# @task(retries=2, retry_delay_seconds=5)
def plot_sleep_score(data, goal=7.5):
    pd.options.plotting.backend = "plotly"
    fig, ax1 = p.subplots()

    sleep_df = get_sleep_df(data)
    sc = get_sleep_score(sleep_df, goal)

    ax1.bar(sleep_df.index, sleep_df.deepsleep + sleep_df.rem + sleep_df.lightsleep,
            label='total sleep')
    ax1.bar(sleep_df.index, sleep_df.deepsleep + sleep_df.rem,
            label='deepsleep + rem')
    ax1.bar(sleep_df.index, sleep_df.rem, label='rem')

    if len(sleep_df) < 10:  # if less than 10 days, then print durations
        y_offset = -0.5
        for patch in ax1.patches:
            b = patch.get_bbox()
            duration = p.ceil((b.y1 - b.y0) * 60)
            minutes, hours = int(duration % 60), int(duration // 60)
            val = "{:d}h {:02d}min".format(hours, minutes)
            ax1.annotate(val, ((b.x0 + b.x1) / 2, b.y1 + y_offset),
                         ha='center')

    ax1.axhline(goal, c='m', ls='--', alpha=0.3)
    ax1.set_ylabel('duration (h)')
    ax1.legend(loc='upper left')

    ###TODO writes on previous chart
    # ax2 = ax1.twinx()
    # #
    # k = ['lightsleep', 'deepsleep', 'rem']
    # normalized_durations = sleep_df[k]
    # normalized_durations = normalized_durations.div(normalized_durations.sum(1), 0)*100
    #
    # ax2.plot(sleep_df.index, sc.total,  color='C4',
    #          label='sleep score')
    # ax2.plot(normalized_durations.index,
    #          normalized_durations.deepsleep + normalized_durations.rem,
    #           color='C2', label='% (deepsleep + rem)')
    # ax2.plot(normalized_durations.index,
    #          normalized_durations.rem,
    #           color='C3', label='% rem')
    # ax2.set_ylim(0, 100)
    # ax2.legend(loc='upper right')
    return fig


######### PLOTTING FUNCTIONS FOR EMOTIONS - HABITS - TASKS DATA
def emotions_timeline():
    """# trying to visualise
    try to astype linux datetime format for x for go.Violin"""

    df_emotions1 = emotions_habits_df_prepare()[2]

    fig7 = go.Figure()
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 30, colortype='rgb')

    for i, color in zip(df_emotions1.iloc[:, 2:], colors):
        # y=df_emotions1.loc[:,i],
        # fig.add_trace(go.Violin(x=df_emotions1["datetime_rnd"][109:].astype(str), y=np.array(df_emotions1.loc[109:,i]),
        #                         line_color=color))#, name=i

        fig7.add_trace(go.Scatter(x=df_emotions1["datetime_rnd"], y=df_emotions1.loc[:, i], name=i, mode="lines",
                                  line_shape='spline'))  # , line_color=color

    fig7.update_layout(
        title="Emotions timeline", xaxis_title="Date", yaxis_title="amount", template="plotly_dark",
        yaxis_range=[0, 1.5]
    )
    # fig.update_traces(orientation='h', side='positive', width=2, points=False)
    # fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    # fig7.show()
    return fig7


def emotions_radial_timeline():
    """ sources of inspiration
    https://medium.com/@marcosanchezayala/plotting-pokemon-attributes-plotly-polar-plots-and-animations-319934b60f0e
    https://stackoverflow.com/questions/71781424/i-want-to-make-an-animated-polar-chart-but-the-chart-i-get-only-has-one-radii-wi"""

    df_emotions1 = emotions_habits_df_prepare()[2]

    # preparing DFs
    df_emotions2 = df_emotions1.drop(columns=['act'])
    # df_emotions2
    df_emotions2 = pd.melt(df_emotions2, id_vars='datetime_rnd', var_name='attribute', value_name='attribute_value')
    df_emotions2['datetime_rnd'] = df_emotions2['datetime_rnd'].astype(str)
    df_emotions2['attribute_value'] = df_emotions2['attribute_value'] + 0.7

    fig8 = px.line_polar(df_emotions2,
                         r='attribute_value',
                         theta='attribute',
                         line_close=True, line_shape='spline',
                         animation_frame='datetime_rnd',
                         title=f"{random_emoji()}" + ' Emotions',
                         range_r=(0, 2),
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template="plotly_dark",
                         )
    fig8.update_layout(
        autosize=False,
        height=500,
        width=750,
    )

    # fig8.show()
    # pyo.plot(fig)
    # fig8.write_html('plot_emotions.html')
    return fig8


def habits_radial_timeline():
    """PLOTTING ANIMATED TIMESERIES FOR habits ARCHIVE

    # inspiration https://medium.com/@marcosanchezayala/plotting-pokemon-attributes-plotly-polar-plots-and-animations-319934b60f0e
    # https://stackoverflow.com/questions/71781424/i-want-to-make-an-animated-polar-chart-but-the-chart-i-get-only-has-one-radii-wi

    # one-hot encoding for trying get data for plotting timeseries on radial plot
    # getting dummies from list from grouped by rounded 10m datetime periods
    # https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies"""

    df_habits = emotions_habits_df_prepare()[1]

    df_habits1 = df_habits.groupby('datetime_rnd')['act'].agg(list).reset_index()  # .iloc[1,:]

    # GENERATING getdummies from list of emotions (act column) grouped by 10min periods
    df_habits1 = df_habits1.join(df_habits1['act'].str.join('|').str.get_dummies())
    # df_habits1

    # preparing DFs
    df_habits2 = df_habits1.drop(columns=['act'])
    # df_emotions2
    df_habits2 = pd.melt(df_habits2, id_vars='datetime_rnd', var_name='attribute', value_name='attribute_value')
    df_habits2['datetime_rnd'] = df_habits2['datetime_rnd'].astype(str)
    # adding 1 just for  better look
    df_habits2['attribute_value'] = df_habits2['attribute_value'] + 0.7

    fig9 = px.line_polar(df_habits2,
                         r='attribute_value',
                         theta='attribute',
                         line_close=True, line_shape='spline',
                         animation_frame='datetime_rnd',
                         title=f"{random_emoji()}" + ' Habits',
                         range_r=(0, 2),
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template="plotly_dark",
                         )
    fig9.update_layout(
        autosize=False,
        height=500,
        width=750#, margin=dict(t=15, l=15, b=15, r=15)
    )

    # fig9.show()
    # pyo.plot(fig)
    # fig9.write_html('plot_habitts.html')
    return fig9


def interesting_numbers(df_tasks_3_t, df_sessions):
    """####### INTERESTING NUMBERS FROM DATA
        # week- month - dayly(main) statictics

        #area balance
        # number of tasks completed in a week
        # emotional balance

        #habits balance
        #creativity tasks??
        # tasks difficulty
        # amount of tracked time per timeperiod
        #amount of money earned per timeperiod

        #amount of stress per timeperiod

        #recommended tasks to do
        """

    # completed tasks
    number_tasks_per_day(df_tasks_3_t).iloc[:, :1]
    number_tasks_per_day_per_task(df_tasks_3_t)
    number_tasks_per_day_per_task(df_tasks_3_t)
    number_hard_tasks_per_m(df_tasks_3_t).iloc[:, :1]

    # TODO WTFFFFF
    check_goals_difficulty(df_tasks_3_t)
    tasks_done_per_month(df_tasks_3_t).iloc[:, :1]
    amount_tasks_done_life_areas(df_tasks_3_t).iloc[:, :1]
    amount_complited_tasks_permonth(df_tasks_3_t).iloc[:, :1]
    amount_complited_tasks_per_list_month(df_tasks_3_t, 'BACKLOG life').iloc[:, :1]

    # creativity??? sparks
    amount_new_tasks_per_day(df_tasks_3_t)

    # TODO plot and see trend, timewindows????
    df_tasks_3_t[df_tasks_3_t.underrated == False].underrated.count() / df_tasks_3_t[
        df_tasks_3_t.underrated == True].underrated.count()

    df_tasks_3_t[df_tasks_3_t.underrated==False]
    df_tasks_3_t[df_tasks_3_t.metricks.notnull()]
    df_tasks_3_t[(df_tasks_3_t.underrated==False) & (df_tasks_3_t.status == 'NotStarted')]

    # TODO fix names of tasks
    df_sessions.groupby(['subject']).agg({'duration': sum})


# @st.cache(allow_output_mutation=True)
def ideas_sparks(df_tasks_3_t):
    """#creativity - ideas sparks
    amount of new ideas-tasks written down on TO DO app"""

    from dateutil.relativedelta import relativedelta

    df_new_tasks_pday = amount_new_tasks_per_day(df_tasks_3_t)

    left_range = df_new_tasks_pday.created_datetime.max() + relativedelta(months=-1)
    right_range = df_new_tasks_pday.created_datetime.max()
    fig16 = px.line(df_new_tasks_pday,
                    x="created_datetime",
                    y=[df_new_tasks_pday.subject, df_new_tasks_pday.subject.rolling(30).mean()],
                    color_discrete_sequence=px.colors.qualitative.G10,
                    template="plotly_dark", line_shape='spline',
                    range_x=(left_range,right_range),  title='Creativity')

    fig16.update_layout(
                autosize=False,
                height=190,
                width=250,
                # plot_bgcolor="white",
                margin=dict(t=25, l=15, b=0, r=15),
                showlegend=False,
                xaxis_title='',
                yaxis_title=''
        ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    fig16.update_traces(line=dict(width=1.5), opacity=.9,)

    # fig16.update_xaxes(visible=False)
    # fig16.update_yaxes(visible=False)#, fixedrange=True
    # fig16.show()
    return fig16


# @st.cache()
def life_areas_balance_count(df_tasks_3_t, df_habits, period='W'):
    """count habits and tasks done by time period grouping by life_areas
        not counting summary values of habbits, just counting amount of acts """
    from shapely.geometry import MultiPoint
    habits_list_for_check = {
    'PHYSICAL': ['slept_GOOD', 'cold shower', '10 pushups everyday', 'big physical activity',
                              'over_eated', 'fastfood', 'cafe'],

    'FUN_RECREATION': ['traveled', 'drugs', 'jrk', 'porn', 'too much movies',
    'too much youtube', 'too much social media', 'too much news', 'any pain',
    '+youtubed', 'film', 'tvshow', 'cinema', 'gaming'],

    'INTELECTUAL': ['made smth for selfefficiency',
    'how many info i consumed vs generated out in the world', 'running on plans feeling', 'psycho practices',
    '5 minute journal', 'meditation', 'look inside 4 feelings on whole life', 'bucket list',
    'wish list', 'reading', 'studing', 'chess', 'ankicards', 'languages'],

    'LOVE ROMANCE SEX': ['sex','new sex partner', 'new sex practices'],

    'PARTNER': ['harmony_pleasurefull', 'critiqued_by_HER',
    'critiqued_by_ME', 'arguing', 'work thru complicated situations', 'common goals completion'],

    'SOCIAL FRIENDS': ['social offline', 'new people', 'old_friends'],

    'FINANCIAL': ['encreased income','financial reduce costs', 'investing']
    }

    # done tasks by life_areas
    df_done_tasks_by_lifearea = amount_tasks_done_life_areas(df_tasks_3_t, time_period=period).iloc[:,:1].reset_index()\
        .rename(columns={'completed_datetime': 'datetime_rnd', 'subject': 'act'})

    df_habits['life_area'] = ''
    for k in df_habits.act.unique(): # adding column life_area to each acts responsively
        for i in habits_list_for_check.keys():
            if k in habits_list_for_check[i]:
                df_habits.loc[df_habits['act'] == k, 'life_area'] = i
    df_habits = df_habits[df_habits.life_area != '']  # dropping rows which arent in dictionary

    # grouping by period of time and life area and counting similar acts
    df_habits1 = df_habits.groupby([pd.Grouper(key='datetime_rnd', axis=0,
                                               freq=period), df_habits.life_area]).agg({'act': 'value_counts'})

    df_habits1.index.names = ["datetime_rnd", "life_area", "acts"] # renaming duplicate columns
    df_habits1 = df_habits1.reset_index().groupby(['datetime_rnd', 'life_area']).agg({'act': sum}).reset_index()
    # df_habits1

    out = pd.concat([df_done_tasks_by_lifearea, df_habits1])
    out = out.groupby(["datetime_rnd", "life_area"], as_index=False).sum()

    # making mean values for year
    out_mean_year = out.groupby([pd.Grouper(key='datetime_rnd', axis=0,
                                            freq='Y'), out.life_area]).agg({'act': 'mean'}).reset_index()
    # out_mean_year.rename(columns={'act': 'act_mean'}, inplace=True)
    out_mean_year.act_mean = out_mean_year.act.round(0).astype(int)
    out_mean_year.datetime_rnd = out.datetime_rnd.mean()
    out['model'] = 'daily'
    out_mean_year['model'] = 'year_mean'

    out2 = pd.DataFrame()
    temp = out_mean_year.copy()  # df to copy on all unique dates to show year mean values constantly
    # print(out.life_area.unique())

    for ik in out.datetime_rnd.unique():
        temp.datetime_rnd = ik
        out2 = pd.concat([out2, temp], axis=0)

    out = pd.concat([out, out2], axis=0)

    # preparing df for plotting cosmetics
    out['datetime_rnd'] = out['datetime_rnd'].astype(str)
    out.datetime_rnd = out.datetime_rnd.astype('datetime64').dt.strftime('%Y-%m-%d')

    out.drop(out[out.life_area.isin(['FUNRECREATION', 'INTELLECTUAL'])].index, inplace=True)

    out['act'] = out['act'] + 0.3

    out.sort_values(by=['life_area', 'act'], inplace=True)

    max_r = out.act.max()

    # print(out.model.value_counts())
    # print(out)

    # https://stackoverflow.com/questions/73624867/how-to-calculate-area-of-a-radar-chart-in-plotly-matplotlib
    # compare areas of two plots
    # convert theta to be in radians
    out["theta_n"] = pd.factorize(out["life_area"])[0]
    out["theta_radian"] = (out["theta_n"] / (out["theta_n"].max() + 1)) * 2 * np.pi
    # work out x,y co-ordinates
    out["x"] = np.cos(out["theta_radian"]) * out["act"]
    out["y"] = np.sin(out["theta_radian"]) * out["act"]
    df_a = out.groupby("model").apply(lambda d: MultiPoint(list(zip(d["x"], d["y"]))).convex_hull.area)

    # print(df_a)

    # print(out[out.datetime_rnd > '2022-07-24'])

    fig12 = px.line_polar(out[out.datetime_rnd > '2022-07-24'],
                          r='act',
                          theta='life_area', color="model",
                          line_close=True,  line_shape='spline',
                          range_r=(0, max_r),
                          animation_frame='datetime_rnd',
                          title=f"{random_emoji()}" + 'Emotions Weekly Balance',
                          color_discrete_sequence=px.colors.sequential.Plasma_r,
                          template="plotly_dark",

                 )
    fig12.update_layout(
        autosize=False,
        height=550,
        width=650,
        # showlegend=False,
        margin=dict(t=45, l=45, b=15, r=15),
        xaxis_title='',
        yaxis_title=''
    ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})

    # fig12.show()
    # pyo.plot(fig)
    # fig12.write_html('plot_life_areas_balance.html')
    return fig12


# @st.cache()
def working_time_stats(df_sessions):
    """plotting 3 stats on everyday working sessions
    number of sessions per day
    amount of hours tracked per day
    average for 30days for daily workhours
    """
    from dateutil.relativedelta import relativedelta

    df_sessions['for_count'] = df_sessions['subject'].map(type) == str

    # number of sessions per day
    number_sessions_per_day = df_sessions.groupby(pd.Grouper(key='start', axis=0,
                                         freq='D')).agg({'for_count':sum}).reset_index()

    # amount of hours tracked per day
    amount_of_hours_tracked_per_day = df_sessions.groupby([pd.Grouper(key='start', axis=0,
                                         freq='D'),]).agg({'duration':sum})/ 3600
    # number_sessions_per_day.start = number_sessions_per_day.start.astype(str)

    out=pd.concat([number_sessions_per_day.set_index('start'),amount_of_hours_tracked_per_day], axis=1).reset_index()

    left_range = out.start.max() + relativedelta(months=-1)
    right_range = out.start.max()

    fig19 = px.line(out, x=out.start, y=[out.for_count, out.duration, out.duration.rolling(30).mean()],
                            line_shape = 'spline',
                            title='Amount of Sessions',
                            color_discrete_sequence=px.colors.sequential.Plasma_r,
                            template="plotly_dark", range_x=(left_range,right_range)
                     )
    fig19.update_layout(
            autosize=False,
            height=190,
            width=250,
            showlegend=False,
            # plot_bgcolor="white",
            margin=dict(t=25, l=15, b=0, r=15),
            xaxis_title='',
            yaxis_title=''
        ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})

    # fig19.show()

    return fig19


# @st.cache()
def number_done_tasks_per_d(df_tasks_3_t):
    from dateutil.relativedelta import relativedelta

    number_done_tasks_per_day = number_tasks_per_day(df_tasks_3_t).iloc[:,:1].reset_index()

    left_range = number_done_tasks_per_day.completed_datetime.max() + relativedelta(months=-1)
    right_range = number_done_tasks_per_day.completed_datetime.max()

    fig18 = px.line(number_done_tasks_per_day, x="completed_datetime",y=[number_done_tasks_per_day.subject,
                                                                       number_done_tasks_per_day.subject.rolling(30).mean()],
                  color_discrete_sequence=px.colors.sequential.Plasma_r,
                  template="plotly_dark", line_shape='spline', range_x=(left_range,
                                                                        right_range), title='Number Done Tasks Per Day')

    fig18.update_layout(
            autosize=False,
            height=190,
            width=250,
            showlegend=False,
            # plot_bgcolor="white",
            margin=dict(t=25, l=15, b=0, r=15),
            xaxis_title='',
            yaxis_title=''
        ).update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)"})
    # fig18.show()
    return fig18


########## STREAMLIT DASHBOARD #####################

# @task(retries=2, retry_delay_seconds=15)
# @flow(name='streamlit_dashboard')
def streamlit_dash():
    global df_emotions, df_habits, df_emotions1, df_tasks_3, df_tasks_3_t

    #####  CONNECT to DBs ########
    # DB_ADASH1, DB_TODO, DB_ACTWATCH, DB_GADGETBRIDGE, DB_ADASH2, DB_ADASH3, DB_ADASH4, DB_EMOTIONS_TEST = read_db_paths()
    df_emotions, df_habits, df_emotions1 = emotions_habits_df_prepare()

    #####  MANIPULATE DATAFRAMES ########
    df_sessions = pd.DataFrame(sessions_download())
    df_sessions.rename(columns={'description': 'subject'}, inplace=True)
    df_sessions = df_sessions[df_sessions.duration > 0]

    DB_ADASH1 = "./android_db/com.actiondash.playstore/databases/app_info"
    DB_ADASH2 = "./android_db/com.actiondash.playstore/databases/usage_events"
    DB_ADASH3 = "./android_db/com.actiondash.playstore/databases/UsageStatsDatabase"
    DB_ADASH4 = "./android_db/com.actiondash.playstore/databases/NotificationEntity"
    DB_ACTWATCH = "./android_db/net.activitywatch.android/files/sqlite.db"
    DB_TODO = "./android_db/todosqlite.db"
    DB_GADGETBRIDGE = "./android_db/nodomain.freeyourgadget.gadgetbridge/databases/Gadgetbridge"
    DB_EMOTIONS_TEST = "./android_db/habits_emotions.csv"

    df_tasks, df_joined = df_tasks_prepare(DB_TODO)
    df_tasks_2 = tasks_read_metrics(df_tasks, df_joined)
    df_tasks_3 = tasks_motivation(df_tasks_2)

    df_tasks_3_t = after_features_assumpted(df=df_sessions, df1=df_tasks_3, df_tasks_3=df_tasks_3,
                                            df_emotions1=df_emotions1)

    #######   PREPARATION DATAFRAMES FOR ML ################

    # df_done_tasks_5_ml = prepare_df_for_ml(df_tasks_3_t)[0]  # this one is for training
    # df_not_started_tasks_4_ml_prediction = prepare_df_for_ml(df_tasks_3_t)[1]  # this one is for prediction
    df_final = prepare_data_tasks_sessions_emohabits(DB_EMOTIONS_TEST, df_tasks_3_t, df_sessions)
    # df_hab_emo_for_ML = prepare_emohab_2_for_ML_knn(df_final)

    #######   PREDICTION OF KNN_CLASSIFIER for tasks  ################

    # X, y, X_train, X_test, y_train, y_test = define_xy_for_knn(df_done_tasks_5_ml, target='TTC_aft')

    # best_k = knn_clf(X_train, X_test, y_train, y_test, detect_best_k=0)[0]
    # best_accuracy = knn_clf(X_train, X_test, y_train, y_test, detect_best_k=0)[1]
    #
    # save_load_predict_model_knn(X_train, X_test, y_train, y_test, best_k, best_accuracy, X_test, target='TTC_aft',
    #                             train_model=1, save_model_file=1, load_model_and_predict=1)

    # get predictions
    # df_not_started_tasks_4_ml_prediction['TTC_pred'] = save_load_predict_model_knn(X_train, X_test, y_train, y_test,
    #                                                                                best_k=2,
    #                                                                                best_accuracy=0.5384615384615384,
    #                                                                                target='TTC_aft',
    #                                                                                df_to_predict=df_not_started_tasks_4_ml_prediction,
    #                                                                                train_model=0, save_model_file=0,
    #                                                                                load_model_and_predict=1)
    # knn_from_joblib = joblib.load('TTC_.pkl')

    # Use the loaded model to make predictions
    # modelscorev23.predict_proba(x_test)
    # df_not_started_tasks_4_ml_prediction['TTC_pred'] = knn_from_joblib.predict(df_not_started_tasks_4_ml_prediction)

    # # sorting and getting index of tasks to get task names back
    # df_not_started_tasks_4_ml_prediction.sort_values(by=['TTC_pred'], inplace=True, ascending=False)  # .index
    # indexes = df_not_started_tasks_4_ml_prediction.index
    #
    # # list of names of tasks
    # list_knn_tasks = df_tasks_3.iloc[indexes, 0].to_list()

    # TODO add to dashboard knn predictions for sessions amount
    # problem - prediction must be very synced, very fast. like every hour??
    # knn_predict_sessions_amount_based_on_emotions(df_final)

    # TODO add to dashboard timeseries predictions
    # predict_timeseries_emohab(df_final)

    # TODO add logic for adding REMINDERS for Gcalendar
    # add_popup_reminder(name: str, calendar_name, start_time)

    # TODO refactor to one type figure call
    functions = (
        plot_mean_heart_rate,
        plot_mean_raw_intensity,
        plot_steps_boxplot_by_weekday,
        plot_heart_rate_boxplot_by_weekday,
        plot_activity_boxplot_by_weekday,
        plot_mean_week,

        # plot_cumstep_by_day,
        # plot_calendar_of_steps,
        # plot_calendar_of_activity,
        # plot_calendar_of_sleep_duration,
        # plot_calendar_of_deepsleep_percentage,
        # plot_calendar_of_sleep_score,
        # plot_sleep_score,
    )

    for f in functions:
        f(data)

    ###### CONFIGURING DASHBOARD GRID AND LAYOUT ITEMS #######
    st.set_page_config(layout="wide")
    st.header("THIS DASHBOARD IS " + f"{random_emoji()}")
    st.write("lot of functionality is cut due to sec reasons. data is anonimised." + f"{random_emoji()}")

    def make_grid(cols, rows):
        """ make any grid with a function
         https://towardsdatascience.com/how-to-create-a-grid-layout-in-streamlit-7aff16b94508"""

        grid = [0] * cols
        for i in range(cols):
            with st.container():
                grid[i] = st.columns(rows, gap="small")
        return grid

    mygrid = make_grid(1, 5)  # row and col respectively
    mygrid[0][0].plotly_chart(life_areas_balance_count(df_tasks_3_t, df_habits, period='W'))

    last_month_amount_done_tasks, how_changed, today_creat, how_changd_2 = done_tasks_and__creativity_simple_metricks(df_tasks_3_t)

    from PIL import Image
    image = Image.open('image.png')
    mygrid[0][3].image(image)

    mygrid[0][3].metric(f"{random_emoji()}" + "My monthly amount done tasks", str(last_month_amount_done_tasks),
                        str(how_changed)+'%')
    mygrid[0][3].metric(f"{random_emoji()}" + "My daily creativity", today_creat, str(how_changd_2)+'%')

    #  adding random correlated emotions and habits with corr value as a metric
    input_str_for_correlated = str(print_highly_correlated(df=df_final, features=df_final.columns))
    digits_here = int(re.findall(r'\d+', input_str_for_correlated)[1])
    digits_here_int = int(digits_here//10)
    leave_only_letters = (re.sub(r'[^ \w+]', '', input_str_for_correlated)).split(' ')
    mygrid[0][3].write(f"{random_emoji()}" + 'think why these 2 are correlated:', )

    put__string = '<p class="big-font" style="margin-top: -28px;">' + leave_only_letters[0] + '</p>' \
                  + '<p class="big-font"style="margin-top: -28px;margin-bottom: -38px;"">' + \
                  leave_only_letters[2] + '</p>'
    mygrid[0][3].markdown("""<style>.big-font {font-size:32px !important;font-weight: 400 !important;}</style>""",
                          unsafe_allow_html=True)
    mygrid[0][3].markdown(put__string, unsafe_allow_html=True)

    mygrid[0][3].metric(label="", value='', delta=str(digits_here_int) + '%')

    mygrid[0][4].plotly_chart(ideas_sparks(df_tasks_3_t))
    mygrid[0][4].plotly_chart(working_time_stats(df_sessions))
    mygrid[0][4].plotly_chart(number_done_tasks_per_d(df_tasks_3_t))

    # TODO fix to plot
    # mygrid[0][1].plotly_chart(correlation_matrix_for_emohabits(df_final)) #, height=100)
    #

    mygrid1 = make_grid(2, 5) # row and col respectively
    # mygrid1[0][0].plotly_chart(plot_cumstep_by_day(data)[0], use_container_width=True)
    # mygrid1[0][1].plotly_chart(plot_cumstep_by_day(data)[1], use_container_width=True)
    # mygrid1[0][2].plotly_chart(fig2, use_container_width=True)
    # mygrid1[0][3].plotly_chart(fig3, use_container_width=True)

    # mygrid1[0][4].table(count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[0])

    #####  COLD START + KNNed tasks as lists  ########
    # mygrid1[0][1].header('AI tells u to do')
    # mygrid1[0][1].subheader('underrated tasks')
    # mygrid1[0][2].subheader('active tasks')
    # mygrid1[0][3].subheader('knned tasks')

    # mygrid1[0][1].table(set(cold_start_ml(df_tasks_3_t)[1]))
    # mygrid1[0][2].table(cold_start_ml(df_tasks_3_t)[0])
    # mygrid1[0][3].table(list_knn_tasks)

    # mygrid1[1][0].plotly_chart(emotions_timeline())
    mygrid1[1][0].plotly_chart(emotions_radial_timeline())
    mygrid1[1][2].plotly_chart(habits_radial_timeline())

    ######## MAKING FORM FOR DYNAMICALLY SENDing CHOSEN TASKS ON BUTTON CLICK TO GCAL
    st.header(f"{random_emoji()}" + ' THIS MACHINE TELLS U TO DO SO:')
    st.write(".                "+f"{random_emoji()}" + ' Behave well stupid human')

    # dynamically generating names for each task
    def make_tasks_names(list_of, prefix):
        return [f'{prefix}_{e}' for e in range(len(list_of))]

    list_task1 = set(cold_start_ml(df_tasks_3_t)[1][:10])
    list11 = make_tasks_names(list_task1, 'cold_1')
    ranged_tasks_dict1 = dict(zip(list11, list_task1))

    list22 = make_tasks_names(set(cold_start_ml(df_tasks_3_t)[0][:10]), 'cold_2')
    ranged_tasks_dict2 = dict(zip(list22, set(cold_start_ml(df_tasks_3_t)[0][:10])))

    # list33 = make_tasks_names(list_knn_tasks[:6], 'knned_1')
    # knned_tasks_dict = dict(zip(list33, set(list_knn_tasks[:6])))

    merged_dict = ranged_tasks_dict1 | ranged_tasks_dict2 #| knned_tasks_dict

    with st.form(key='columns_in_form'):
        colms = st.columns(3)
        colms[0].subheader(f"{random_emoji()}" + 'underrated tasks')
        colms[0].write('feature was cut out in demonstration version ')
        colms[1].subheader(f"{random_emoji()}" + 'active tasks')

        colms[2].subheader(f"{random_emoji()}" + 'knned tasks')
        colms[2].write('feature was cut out in demonstration version ')

        for idd, title in ranged_tasks_dict1.items():
            globals()[idd] = colms[0].checkbox(f'{title}', key=idd)

        for idd, title in ranged_tasks_dict2.items():
            globals()[idd] = colms[1].checkbox(f'{title}', key=idd)

        # for idd, title in knned_tasks_dict.items():
        #     globals()[idd] = colms[2].checkbox(f'{title}', key=idd)

        submitted = st.form_submit_button('Submit to Gcalendar  👈')

        tasks_id_to_add_to_gcal = []
        if submitted:
            for ii in list22:
                if globals()[ii]:
                    tasks_id_to_add_to_gcal.append(ii)
            for ii in list11:
                if globals()[ii]:
                    tasks_id_to_add_to_gcal.append(ii)
            # for ii in list33:
            #     if globals()[ii]:
            #         tasks_id_to_add_to_gcal.append(ii)
            # st.write(f'ids {tasks_id_to_add_to_gcal}')

            tasks_titles_to_gcal =[]
            for ko in tasks_id_to_add_to_gcal:
                tasks_titles_to_gcal.append(merged_dict[ko])
            st.write(f'Sending to Gcal {tasks_titles_to_gcal}')

            sessions_calendar = GoogleCalendar(config.get('GCALENDAR', 'sessions'), credentials_path='credentials.json')
            put_tasks1(list_of_tasks_names=tasks_titles_to_gcal, df_tasks11=df_tasks_3_t, calendar_name=sessions_calendar)

    ####### ADDING GCAL IFRAME
    mygrid2 = make_grid(1, 2) # row and col respectively

    with mygrid2[0][0]:
        components.html("""
        <iframe src="https://calendar.google.com/calendar/embed?height=600&wkst=1&bgcolor=%23ffffff&ctz=Asia%2FTbilisi&showTz=0&showCalendars=0&showTabs=1&showPrint=0&showDate=0&showTitle=0&showNav=1&mode=WEEK&src=NTJmZDQ4NDEzYjFkYzE0N2JkMDFjODk1ZmQxZmNhNjQxNTBlODVhOWFkNTMwZThhMmEwNjU0MzNmODhlNGRiMUBncm91cC5jYWxlbmRhci5nb29nbGUuY29t&color=%23616161" style="filter: invert(1) saturate(0.1) hue-rotate(200deg);" width="600" height="600" frameborder="0" scrolling="no"></iframe>
        """, height=700,)

    mygrid2[0][1].write(f"{random_emoji()}" + 'Mr Gantt corner')
    mygrid2[0][1].plotly_chart(draw_gant(df_sessions)) # add gant diagram

    # ###### ADDING ACTIVITYWATCH IFRAME
    # components.iframe("http://localhost:5600/#/timeline/", width=1200,
    #                   height=400, scrolling=True)

    ####### DATA FROM SMARTWATCH

    col3 = make_grid(7, 3)

    col3[0][0].plotly_chart(fig6, use_container_width=True)
    col3[0][1].plotly_chart(fig2, use_container_width=True)
    col3[0][2].plotly_chart(plot_cumstep_by_day(data)[0], use_container_width=True)
    col3[0][2].plotly_chart(plot_cumstep_by_day(data)[1], use_container_width=True)

    col3[1][0].plotly_chart(fig3, use_container_width=True)
    col3[1][1].plotly_chart(fig4, use_container_width=True)
    col3[1][2].plotly_chart(fig5, use_container_width=True)

    col3[3][0].plotly_chart(plot_calendar_of_steps(data), use_container_width=True)
    col3[3][1].plotly_chart(plot_calendar_of_activity(data), use_container_width=True)
    col3[3][2].plotly_chart(plot_calendar_of_sleep_duration(data), use_container_width=True)

    col3[4][0].plotly_chart(plot_calendar_of_deepsleep_percentage(data), use_container_width=True)
    col3[4][1].plotly_chart(plot_calendar_of_sleep_score(data), use_container_width=True)
    col3[4][2].plotly_chart(plot_sleep_score(data), use_container_width=True)



########## DASH DASHBOARD
# app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

# app.layout = html.Div([
#     dbc.Row(dbc.Col(html.H1(children='FUCKING awesome'))),
#
#     html.Div([dcc.Graph(figure=fig1),
#               dcc.Graph(figure=fig2),
#               dcc.Graph(figure=fig3),
#               dcc.Graph(figure=fig4),
#               dcc.Graph(figure=fig5),
#               dcc.Graph(figure=fig6),
#               dcc.Graph(figure=fig7)
#               ]),
#
#     # html.Div([dcc.Graph(id='example-graph', figure=fig6), dcc.Graph( figure=fig3)])
#
# ])
#
# app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter


if __name__ == "__main__":
    streamlit_dash()
