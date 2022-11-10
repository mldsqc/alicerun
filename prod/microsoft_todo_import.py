from O365 import Account, MSOffice365Protocol, FileSystemTokenBackend
import datetime

import psycopg2
from python_postgresql_dbconfig import read_db_config

credentials = ('f2059297-b41e-4e94-9e98-5dbc982ed0df', 'Ssw8Q~jlbVDpxMX4THbOA1-PYLf8po7~5uiwybGX')
token_backend = FileSystemTokenBackend(token_filename='/src/Telegram_bot/o365_token.txt')

#secret_value = Ssw8Q~jlbVDpxMX4THbOA1-PYLf8po7~5uiwybGX
#secret_id=baceca2b-dbb3-4c95-831c-e511854d3b35
account = Account(credentials, protocol=MSOffice365Protocol(), token_backend=token_backend)

if not account.is_authenticated:  # will check if there is a token and has not expired
    # ask for a login
    # console based authentication See Authentication for other flows
    account.authenticate(scopes=['basic', 'tasks_all'])

# if account.authenticate(scopes=['basic', 'tasks_all']):
#    print('Authenticated!')

# try the api version beta of the Microsoft Graph endpoint.
# protocol = MSGraphProtocol(api_version='beta')  # MSGraphProtocol defaults to v1.0 api version
# account = Account(credentials, protocol=protocol)

# protocol_graph = MSGraphProtocol()
# scopes_graph = protocol.get_scopes_for('tasks_all')



# m = account.new_message()
# m.to.add('to_example@example.com')
# m.subject = 'Testing!'
# m.body = "George Best quote: I've stopped drinking, but only while I'm asleep."
# m.send()


# # scopes here are: ['https://graph.microsoft.com/Mail.ReadWrite', 'https://graph.microsoft.com/Mail.Send']
#
# account = Account(credentials, scopes=scopes_graph)



# ...

#
# #list current tasks
# folder = todo.get_default_folder()
# print(folder)
# new_task = folder.new_task()  # creates a new unsaved task
# new_task.subject = 'Send contract to George Best'
# new_task.due = dt.datetime(2020, 9, 25, 18, 30)
# new_task.save()
#
# #some time later....
#
# new_task.mark_completed()
# new_task.save()
#
# # naive datetimes will automatically be converted to timezone aware datetime
# #  objects using the local timezone detected or the protocol provided timezone
# #  as with the Calendar functionality
# new_folder = todo.new_folder('Defenders')

#rename a folder
# folder = todo.get_folder(folder_name='digital_ME')

# todo = account.tasks()
#
# print(todo)

""" Connect to Postgresql database """
db_config = read_db_config()
conn = None
try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    todo = account.tasks()
    folders = todo.list_folders()

    for folder in folders:
        # print(folder, '\n\n -----', )
        for task in folder.get_tasks():
            task_body = task.body
            task_created = task.created
            task_modified = task.modified
            task_completed = task.completed
            task_importance = task.importance
            task_due = task.due
            task_starred = task.is_starred

            # print(task, 'Created:', task.created, '', task.body, task.is_starred, task.importance, task.due, task.modified,
            #       task.completed, '\n\n -----', )

            sql = "INSERT INTO todos (task, task_created, task_body, task_modified, task_completed, task_importance, task_due, task_starred) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (str(task), str(task.created), str(task.body), str(task_modified), str(task_completed), str(task_importance), str(task_due),str(task_starred)))

    conn.commit()

except psycopg2.OperationalError as error:
    print("Tasks not added   :(   \n\n  ")
    # print(error)
finally:
    if conn is not None:
        conn.close()
        print('Connection closed | Tasks synced! ')

# folder.name = 'Forwards'
# folder.update()

#list current tasks
# task_list = folder.get_tasks()
# for task in task_list:
#     print(task)
#     print('')

#
# CREATE TABLE todos(
#        task TEXT,
#        task_created TEXT,
#        task_body TEXT,
#        task_completed TEXT,
#        task_importance TEXT,
#        task_due TEXT,
#        task_modified TEXT,
#        task_starred TEXT
#     )
