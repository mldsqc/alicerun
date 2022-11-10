import time

import pyminizip
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

# For using listdir()
import os
import glob

from datetime import datetime, timedelta

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

psswrd=config.get('ZIPFILE', 'pss')
currTS = datetime.now().strftime("%Y%m%d%H")
paths_of_synq_files = [config.get('ZIPFILE', 'activitywatch_pc_path'), config.get('ZIPFILE', 'todosql_path')]


##### AUTHORISING TO GOOGLE DRIVE
gauth = GoogleAuth()

# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")

if gauth.credentials is None:
    # Authenticate if they're not there

    # This is what solved the issues:
    gauth.GetFlow()
    gauth.flow.params.update({'access_type': 'offline'})
    gauth.flow.params.update({'approval_prompt': 'force'})

    gauth.LocalWebserverAuth()

elif gauth.access_token_expired:

    # Refresh them if expired

    gauth.Refresh()
else:

    # Initialize the saved creds

    gauth.Authorize()

# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)


def file_upload_file_drive(path):
    """find 7z file in path and upload to gdrive folder"""

    for x in os.listdir(os.path.dirname(path)):
        if x.endswith('.7z'):

            path = os.path.dirname(path) + '/' + x
            name = os.path.basename(path)

            f = drive.CreateFile({'title': name, 'parents': [{'id': '1fFGIlyWjXfcIaMl-l-YEENnCR_fqAxxo'}]})  # folder id DB_sync
            f.SetContentFile(path)
            f.Upload()

            # Due to a known bug in pydrive if we
            # don't empty the variable used to
            # upload the files to Google Drive the
            # file stays open in memory and causes a
            # memory leak, therefore preventing its
            # deletion
            f = None


def folder_upload_drive(path):
    """# iterating thought all the files/folder
        # of the desired directory"""

    for x in os.listdir(path):

        f = drive.CreateFile({'title': x, 'parents': ["1fFGIlyWjXfcIaMl-l-YEENnCR_fqAxxo"]}) #folder id DB_sync
        f.SetContentFile(os.path.join(path, x))
        f.Upload()

        # Due to a known bug in pydrive if we
        # don't empty the variable used to
        # upload the files to Google Drive the
        # file stays open in memory and causes a
        # memory leak, therefore preventing its
        # deletion
        f = None


def delete_file_in_folder(path):

    # Search files with .7z extension in current directory
    for x in os.listdir(os.path.dirname(path)):
        if x.endswith('.7z'):
            os.remove(os.path.dirname(path) + '/' + x)


def download_all_files_from_folder(folder_id):

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()

    for i, file1 in enumerate(sorted(file_list, key=lambda x: x['title']), start=1):
        print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
        file1.GetContentFile(file1['title'])

    # file.GetContentFile('FILE_NAME_AS_YOU_WANT_TO_SAVE.EXTENSION')


def create_zip(path_offile, psswrd):

    name = os.path.basename(path_offile)
    out_path = os.path.dirname(path_offile)+r'//'+name+"_"+currTS+"_"+".7z"
    # print(out_path)
    # print(name)
    pyminizip.compress(path_offile, None, out_path, psswrd, 7)


def uncompress_zip(path, out, psswrd):
    pyminizip.uncompress(path, psswrd, out, 1)


def zip_upload_delete(path):

    create_zip(path, psswrd=psswrd)
    time.sleep(3)
    file_upload_file_drive(path)
    time.sleep(3)
    print("file uploaded")
    delete_file_in_folder(path)
    print("file deleted")





# filepath = r"D:\peewee-sqlite.v2.db"
# zippath = r"D:\LEETCODE-BEST_"+currTS+"_.7z"
# outpath =r'd:\\'


# create_zip(filepath, zippath, 'test')
# uncompress_zip(zippath, outpath, 'test')
# file_upload_file_drive(zippath)
# file_upload_file_drive(activitywatch_pc_path)
