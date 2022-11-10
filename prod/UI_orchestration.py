# # Prefect imports
# from prefect import task, flow
# from prefect.tasks import task_input_hash
import getopt
#
# import numpy as np
# import pylab as p
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
#
# # GRAPH AND DASHBOARD LIBS
# import plotly.express as px
# import plotly.tools as tls
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from plotly.colors import n_colors
#
# import dash
# import dash_bootstrap_components as dbc
# from dash_bootstrap_templates import load_figure_template
# from dash import dcc
# from dash import html
#
# import streamlit as st
from streamlit.web import cli as stcli
# import streamlit_dashboard

# OTHER FLOWS
from recommendation_ML import *
from data_preparing import *
# from bot_answers_analysis import load_emotions_habits_values
# from data_preparing import memory_usage
# from model import loaddata, get_shifts, ActivityType, filtered_serie
# from sleep import get_sleep_df, get_sleep_score

import warnings
warnings.filterwarnings('ignore')


# @flow
def development(dashboard_on_off=0):
    global df_emotions, df_habits, df_emotions1, df_tasks_3, df_tasks_3_t

    #####  CONNECT to DBses ########
    DB_ADASH1, DB_TODO, DB_ACTWATCH, DB_GADGETBRIDGE, DB_ADASH2, DB_ADASH3, DB_ADASH4, DB_EMOTIONS_TEST = read_db_paths()
    df_emotions, df_habits, df_emotions1 = emotions_habits_df_prepare()

    #####  MANIPULATE DATAFRAMES ########
    df_sessions = pd.DataFrame(sessions_download())
    df_sessions.rename(columns={'description': 'subject'}, inplace=True)

    df_tasks, df_joined = df_tasks_prepare(DB_TODO)
    df_tasks_2 = tasks_read_metrics(df_tasks, df_joined)
    df_tasks_3 = tasks_motivation(df_tasks_2)

    df_tasks_3_t = after_features_assumpted(df=df_sessions, df1=df_tasks_3, df_tasks_3=df_tasks_3,
                                            df_emotions1=df_emotions1)

    # interesting_numbers()

    #####  COLD START   ########
    # recommend tasks from most forgotten areas
    for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[0]:
        print(return_tasks_list_by(df_tasks_3_t, iii))

    # recommend tasks from least forgotten areas
    for iii in count_balance_of_life_areas_tasks_habits(df_tasks_3_t)[1]:
        print(iii)
        print(return_tasks_list_by(df_tasks_3_t, iii))


    #####  ML FOR   ########
    prepare_data_tasks_sessions_emohabits(DB_EMOTIONS_TEST, df_tasks_3_t, df_sessions)


    #####  BACKUP DATA ########

    #####  SEND PUSHES TO CALENDAR ########
    #####  gant diagram ########
    #####  calendar ########

    #####  DASHBOARD ########
    # if dashboard_on_off == 1:
    #     streamlit_dash()

#
if __name__ == "__main__":

    """running CLI command with arguments
       no need to run streamlit run pythonnamefile"""

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "cl"

    # Long options
    long_options = ["cloud", "local"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-l", "--local"):
                print("Running Streamlit dashboard locally"
                      "visit   -    localhost:")

                sys.argv = ["streamlit", "run", "streamlit_dashboard.py"]
                sys.exit(stcli.main())

            elif currentArgument in ("-c", "--cloud"):
                print("Running backend in cloud")

                development(dashboard_on_off=0)

                print("")

            elif currentArgument in ("-h", "--help"):
                print("for help and documentation visit "
                       "link"
                       "link"
                       "link")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))





