'''
analysis of the database
'''

import pandas as pd
import csv
import statistics
import matplotlib.pyplot as plt

wd = "job-recommendation/"

apps = pd.read_csv(wd + "apps.tsv", delimiter="\t", usecols=["UserID", "JobID", "WindowID"], engine='python')
users = pd.read_csv(wd + "users.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="", usecols=["UserID"], engine='python')
jobs = pd.read_csv(wd + "jobs.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="", usecols=["JobID"], engine='python')


#add a column of 1s to apps to allow easier filtering of data
ones = [1] * len(apps['JobID'])
apps['Ones'] = ones


#data analysis
number_of_rating = len(apps['JobID']) #1,603,111
number_of_users = len(users['UserID']) #389,708
number_of_jobs = len(jobs['JobID']) #1,092,096
number_active_users = len(apps['UserID'].unique()) #321235
number_active_jobs = len(apps['JobID'].unique()) #365668


#works out average number of applications per active user
apps_per_user = apps.groupby('UserID')['Ones'].count()
average_apps_per_user = statistics.mean(apps_per_user.tolist()) #4.9904618114464485

#works out average number of applications per active job
apps_per_job = apps.groupby('JobID')['Ones'].count()
average_apps_per_job = statistics.mean(apps_per_job.tolist()) #4.384061498408392

#remove users with less than two applications
apps_per_user_df = pd.DataFrame(apps_per_user)
filtered_apps_per_user_df = apps_per_user_df[apps_per_user_df.Ones >= 2]
popular_users = filtered_apps_per_user_df.index.tolist()

number_popular_users = len(popular_users) #212,990

#remove jobs with less than two applications
apps_per_job_df = pd.DataFrame(apps_per_job)
filtered_apps_per_job_df = apps_per_job_df[apps_per_job_df.Ones >= 2]
popular_jobs = filtered_apps_per_job_df.index.tolist()

number_popular_jobs = len(popular_jobs) #206,839

filtered_users_apps = apps[apps.UserID.isin(popular_users)]
filtered_apps = filtered_users_apps[filtered_users_apps.JobID.isin(popular_jobs)]

#works out average number of applications per user in the filtered list
filtered_apps_per_user = filtered_apps.groupby('UserID')['Ones'].count() #6.494473060060852
average_filtered_apps_per_user = statistics.mean(filtered_apps_per_user.tolist())

#works out average number of applications per job in the filitered list
filtered_apps_per_job = filtered_apps.groupby('JobID')['Ones'].count() #6.595119696379914
average_filtered_apps_per_job = statistics.mean(filtered_apps_per_job.tolist())

final_jobs = filtered_apps['JobID'].unique().tolist()
final_users = filtered_apps['UserID'].unique().tolist()

#calculates number of applications per windowID
apps_per_windowid = apps.groupby('WindowID')['Ones'].count()
#WindowID
#1    353582
#2    209656
#3    217568
#4    233893
#5    232433
#6    174145
#7    181834

#Calculating average applications per user and job for each window
window_one = apps[apps.WindowID == 1]
apps_per_user = window_one.groupby('UserID')['Ones'].count()
average_apps_per_user = statistics.mean(apps_per_user.tolist()) #5.575947770138145

apps_per_job = window_one.groupby('JobID')['Ones'].count()
average_apps_per_job = statistics.mean(apps_per_job.tolist()) #4.35376109736126


window_two = apps[apps.WindowID == 2]
apps_per_user = window_two.groupby('UserID')['Ones'].count()
average_apps_per_user = statistics.mean(apps_per_user.tolist()) #4.599030425340557

apps_per_job = window_two.groupby('JobID')['Ones'].count()
average_apps_per_job = statistics.mean(apps_per_job.tolist()) #4.012555023923445


window_three = apps[apps.WindowID == 3]
apps_per_user = window_three.groupby('UserID')['Ones'].count()
average_apps_per_user = statistics.mean(apps_per_user.tolist()) #4.809089100594593

apps_per_job = window_three.groupby('JobID')['Ones'].count()
average_apps_per_job = statistics.mean(apps_per_job.tolist()) #4.376481000945427


window_four = apps[apps.WindowID == 4]
apps_per_user = window_four.groupby('UserID')['Ones'].count()
average_apps_per_user = statistics.mean(apps_per_user.tolist()) #5.2158196374016015

apps_per_job = window_four.groupby('JobID')['Ones'].count()
average_apps_per_job = statistics.mean(apps_per_job.tolist()) #4.476506727401482


window_five = apps[apps.WindowID == 5]
apps_per_user = window_five.groupby('UserID')['Ones'].count()
average_apps_per_user_5 = statistics.mean(apps_per_user.tolist()) #5.280408014902994 

apps_per_job = window_five.groupby('JobID')['Ones'].count()
average_apps_per_job_5 = statistics.mean(apps_per_job.tolist()) #4.56099762563529


window_six = apps[apps.WindowID == 6]
apps_per_user = window_six.groupby('UserID')['Ones'].count()
average_apps_per_user_6 = statistics.mean(apps_per_user.tolist()) #4.677670633108598

apps_per_job = window_six.groupby('JobID')['Ones'].count()
average_apps_per_job_6 = statistics.mean(apps_per_job.tolist()) #4.546154649402182


window_seven = apps[apps.WindowID == 7]
apps_per_user = window_seven.groupby('UserID')['Ones'].count()
average_apps_per_user_7 = statistics.mean(apps_per_user.tolist()) #4.445275638674978

apps_per_job = window_seven.groupby('JobID')['Ones'].count()
average_apps_per_job_7 = statistics.mean(apps_per_job.tolist()) #4.437573213588442
