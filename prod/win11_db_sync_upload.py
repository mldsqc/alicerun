from gdrive_upload_sync import *
import shutil

for kk in paths_of_synq_files:
    zip_upload_delete(kk) # uploading with zipping to gdrive

    # project db folder
    target = r"C:\Users\Mi\PycharmProjects\drafts1\digitalize_task_FLOW\src\android_db"
    shutil.copy(kk, target + '/' + os.path.basename(kk))
