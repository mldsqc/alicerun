from gcsa.event import Event
from gcsa.google_calendar import GoogleCalendar
from gcsa.recurrence import Recurrence, DAILY, SU, SA, MO, TU, WE, TH, FR

import time
from gcsa.reminders import EmailReminder, PopupReminder


# from beautiful_date import Jan, Apr, Feb, Mar,May, Jun, Jul, Aug, Sept, Oct, Nov, Dec, hours, days

from beautiful_date import *
from dateutil.parser import parse as dtparse
from datetime import datetime

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# gc = GoogleCalendar(credentials_path='credentials.json')

calendar = GoogleCalendar(config.get('GCALENDAR', 'sessions'), credentials_path='credentials.json')
main_calendar = GoogleCalendar(config.get('GCALENDAR', 'main'), credentials_path='credentials.json')


def add_to_gcal(desc, strt, stp):
    calendar = GoogleCalendar(config.get('GCALENDAR', 'main'))

    event = Event(desc,
                  start=strt,
                  end=stp)
    calendar.add_event(event)


def add_sessions_to_gcal(calendar_name, name: str, start_time, end_time):
    """time format (19 / Apr / 2019)[9:00]"""

    event = Event(
        name,
        start=start_time,
        end=end_time
        # minutes_before_email_reminder=20

        # recurrence=[
        #     Recurrence.rule(freq=DAILY),
        #     Recurrence.exclude_rule(by_week_day=[SU, SA]),
        #     Recurrence.exclude_times([
        #         (19 / Apr / 2019)[9:00],
        #         (22 / Apr / 2019)[9:00]
        #     ])
        # ],
    )

    calendar_name.add_event(event)
    print('TASK added' + " " + name)


# print(D.today())
# print(D.now())


def find_timeslot_for_task(calendar_name):
    start = D.now()
    end = start + 1 * days

    eventStarts = sorted([i.start for i in calendar_name.get_events(start, end)])
    eventEnds = sorted([i.end for i in calendar_name.get_events(start, end)])

    # print(eventStarts)
    # print(eventEnds)

    gaps = [start - end for (start, end) in zip(eventStarts[1:], eventEnds[:-1])]
    # print(gaps)
    return gaps, eventStarts, eventEnds


    # TODO Add sleep-work time
    # if startTime.hour < 10 or startTime.hour >= 23:
    #     print("NOT IN RANGE")
    # else:
    #     print("out of work time zone")

    # if startTime + duration < eventStarts[0]:
    # #         # A slot is open at the start of the desired window.
    # #         return startTime
    # #
    #     for i, gap in enumerate(gaps):
    #         if gap > duration:
    #             # This means that a gap is bigger than the desired slot duration, and we can "squeeze" a meeting.
    #             # Just after that meeting ends.
    #             # return eventEnds[i]


# find_timeslot_for_task()


def add_popup_reminder(name: str, calendar_name, start_time):
    """
    adding reminders to calendar

    #TODO define list of forgotten habits
    #TODO define time to remind - small timeslots(?)

    :param name: title of reminder
    :param calendar_name: calendar type name
    :param start_time: time of reminder
    :return:
    """

    popup = Event(name,
                  start=start_time,
                  reminders=[
                        PopupReminder(minutes_before_start=1)
                  ])
    calendar_name.add_event(popup)
    print('POPUP added' + " " + name)

# add_popup_reminder('blablalba', main_calendar, D.now()+2*minutes) # just testing


def delete_calendar_tasks(calendar):
    """delete all events in calendar"""

    for ev in calendar:
        calendar.delete_event(ev)


def get_list_tasks_today(calendar):
    """without repetitions"""

    ll = set([e.summary for e in calendar[D.now():D.now()+ 1* days:'updated']])
    return ll


def put_tasks1(list_of_tasks_names, df_tasks11, calendar_name):
    """
    Taking list of predicted task names and pulling them to the Gcalendar
    checking for 1 day window - if there are tasks?
    if there are - getting the free intervals checking their size
    if no enough sized timeslots, putting tasks at the end, after last task

    :param list_of_tasks_names: list of predicted task names
    :param df_tasks11: dataframe to get TTC metric for putting into calendar
    :return:
    """

    def how_many_tasks():
        return len(find_timeslot_for_task(calendar_name)[1])

    for i in set(list_of_tasks_names[:5]): # take first 5 task of sorted tasks
        if '----' not in i:
            num_tasks = how_many_tasks()
            gaps = find_timeslot_for_task(calendar_name)[0]

            if num_tasks == 0: # if schedule in 3 day range ( see find_timeslot_for_task() ) is empty
                if D.today()[11] > D.now():
                    start_task_time = D.today()[11]
                else:
                    start_task_time = D.now() + 10 * minutes

                if df_tasks11[df_tasks11.subject == i].TTC.notna().values.any():   # check if there is TTC metric
                    task_duration = int(df_tasks11.loc[df_tasks11.subject==i].TTC.values[0])
                else:
                    task_duration = 1 # default time period for small not TTC defined task

                if task_duration <= 1:  # if work time of task dont need to be sliced
                    add_sessions_to_gcal(calendar_name, name=i, start_time=start_task_time,
                                         end_time=start_task_time + task_duration * 60 * minutes)
                    time.sleep(5)
                    continue

                elif task_duration > 1:     # if task need to be divided
                    number_of_sprints = int(task_duration) + 1  # transforming to int before that was done by round(0)
                    for kk in range(number_of_sprints): # adding task divided in 1-hour sprints
                        add_sessions_to_gcal(calendar_name, name=i, start_time=start_task_time,
                                             end_time=start_task_time + 70 * kk * minutes)
                        time.sleep(5)
                        continue

            elif num_tasks == 1:  # if schedule is not empty
                add_sessions_to_gcal(calendar_name, name=i,
                                     start_time=find_timeslot_for_task(calendar_name)[2][0] + 10 * minutes,
                                     end_time=find_timeslot_for_task(calendar_name)[2][0] + 70 * minutes)
                time.sleep(5)
                continue

            elif num_tasks > 1:     # if schedule is not empty
                if df_tasks11[df_tasks11.subject == i].TTC.notna().values.any():
                    task_duration = int(df_tasks11.loc[df_tasks11.subject==i].TTC.values[0])
                else:
                    task_duration = 1

                i_ = 0  # number of times one task added
                n_gap = 0  # index of gap

                for gap in gaps:   # search for the right gap for task
                    if gap.seconds/3600 >= task_duration and i_ < 1:
                        print(gap.seconds//3600)
                        print(task_duration)

                        add_sessions_to_gcal(calendar_name,
                                        name=i,
                                        start_time=find_timeslot_for_task(calendar_name)[2][n_gap] + 10 * minutes,
                                        end_time=find_timeslot_for_task(calendar_name)[2][n_gap] + 70 * minutes)
                        time.sleep(2)
                        i_ += 1
                        n_gap += 1
                        continue

                    else: # if no right gap add to the end
                        print('too small gap OR already added')
                        n_gap += 1
                        continue

                if i_ < 1:  # in case task never added to the gaps add it to the end
                    print('adding to the end')

                    add_sessions_to_gcal(calendar_name,
                                         name=i,
                                         start_time=find_timeslot_for_task(calendar_name)[2][-1] + 10 * minutes,
                                         end_time=find_timeslot_for_task(calendar_name)[2][-1] + 70 * minutes)








# for event in main_calendar:
#     print(event)

# print(calendar.get_events(query='te'))

# for i in calendar.get_events(start, end, order_by='updated'):

# events_list = calendar.get_events(start, end, order_by='updated')

# print(i.start, i.end, i.summary, i.event_id )

#
# gc.get_events(query='metrics')


# def findFirstOpenSlot(events, startTime, endTime, duration):
#     # def parseDate(rawDate):
#     #     # Transform the datetime given by the API to a python datetime object.
#     #     return datetime.datetime.strptime(rawDate[:-6] + rawDate[-6:].replace(":", ""), '%Y-%m-%dT%H:%M:%S%z')
#
#     eventStarts = [parseDate(e['start'].get('dateTime', e['start'].get('date'))) for e in events]
#     eventEnds = [parseDate(e['end'].get('dateTime', e['end'].get('date'))) for e in events]
#
#     gaps = [start - end for (start, end) in zip(eventStarts[1:], eventEnds[:-1])]
#
#     if startTime + duration < eventStarts[0]:
#         # A slot is open at the start of the desired window.
#         return startTime
#
#     for i, gap in enumerate(gaps):
#         if gap > duration:
#             # This means that a gap is bigger than the desired slot duration, and we can "squeeze" a meeting.
#             # Just after that meeting ends.
#             return eventEnds[i]
#
#     # If no suitable gaps are found, return none.
#     return None



# def add_sessions_to_gcal(df_sessions):
#     """ADD SESSIONS TO GOOGLE CALENDAR
#     """
#
#     tmfmt = '(%d/%B/%Y)[%H:%M]'
#     for i, j in df_sessions.iterrows():
#         # print(j)
#         # print(type(dtparse(j['stop'])))
#         add_to_gcal(desc=j['description'], strt=dtparse(j['start']), stp=dtparse(j['stop']))
#         # add_to_gcal(desc=j['description'], strt=dt.strftime(dtparse(j['stop']), format=tmfmt),stp=dt.strftime(dtparse(j['stop']),format=tmfmt))
#         # add_to_gcal(desc=j['description'], strt=dt.strptime(j['start'], format),stp=dt.strptime(j['stop'],format))