import json
import requests
from base64 import b64encode
import configparser
import pandas as pd

'''
importing sessions from TOGGL

# SESSIONS INFO
# NUMBER SESSIONS PER TASK
# DIFFICULTY VS TTC VS NUMBER OF SESSIONS
# IMPORT SESSIONS TO GOOGLE CALENDAR


'''


config = configparser.ConfigParser()
config.read('config.ini')
#config.read('/src/prod/config.ini')

authHeader = config.get('toggl', 'TOKEN') + ":" + "api_token"
data = requests.get('https://api.track.toggl.com/api/v9/me/time_entries',
                    headers={'content-type': 'application/json','Authorization' : 'Basic %s' %  b64encode(authHeader
                                                                                                          .encode())
                    .decode("ascii")})

df_sessions = pd.read_json(json.dumps(data.json()), orient ='records')
# print(df_sessions)












