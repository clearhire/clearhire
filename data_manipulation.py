#resave database-explanation 
import pandas as pd
import csv
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
import re


wd = "job-recommendation/"

def create_dataframe():
    '''
        Filters out all jobs and users who have fewer than 14 applications.
        Then creates a h5 file of a dataframe with the remaining applications.
        We add a column called 'Data' to the dataframe to help with later processing.
    '''
    apps_df = pd.read_csv(wd + "apps5.tsv", delimiter="\t", usecols=["UserID", "JobID"], engine='python')
    
    alpha = 40
    data = [alpha] * len(apps_df['JobID'])
    apps_df['Data'] = data

    popularity_thres = 14

    df_jobs_cnt = pd.DataFrame(apps_df.groupby('JobID').size(), columns=['count'])
    popular_jobs = list(set(df_jobs_cnt.query('count >= @popularity_thres').index))  # noqa
    jobs_filtered = apps_df[apps_df.JobID.isin(popular_jobs)]

    df_users_cnt = pd.DataFrame(jobs_filtered.groupby('UserID').size(), columns=['count'])
    popular_users = list(set(df_users_cnt.query('count >= @popularity_thres').index))  # noqa
    df_apps_filtered = jobs_filtered[jobs_filtered.UserID.isin(popular_users)]

    print('shape of original ratings data: ', apps_df.shape)
    print('shape of ratings data after dropping unpopular movies: ', jobs_filtered.shape)
    print('shape of ratings data after dropping unpopular jobs and users: ', df_apps_filtered.shape)
    
    '''
    With popularity_thres = 3    
           shape of original ratings data:  (232433, 3)
           shape of ratings data after dropping unpopular movies:  (192445, 3)
           shape of ratings data after dropping unpopular jobs and users:  (164847, 3)

    With popularity thres = 4
           shape of ratings data after dropping unpopular movies:  (177976, 3)
           shape of ratings data after dropping unpopular jobs and users:  (140061, 3)

    With popularity thres = 5
           shape of ratings data after dropping unpopular movies:  (165772, 3)
           shape of ratings data after dropping unpopular jobs and users:  (120283, 3)

    With popularity thres = 14
           shape of ratings data after dropping unpopular movies:  (101994, 3)
           shape of ratings data after dropping unpopular jobs and users:  (40135, 3)
    '''

    df_apps_filtered.to_hdf('stored-data/apps-filtered.h5', key='df', mode='w')
    


def load_data():
    '''
    Was used to save npz files of 
        - a user-job matrix where the rows represent users, the columns represent jobs, a 1 in 
          entry (i,j) that the user i applied to job j, and there are zeros elsewhere
        - a job-user matrix which is the transpose of the user-job matrix
    This is because of the memory quota on Heroku meaning we cannot store these matrices on loading the site.
    Outputs:
        job_hashmap: a map from JobID to the row index in the job-user matrix
        user-hashmap: a map from UserID to the row index in the user-job matrix
        job-ids: a list of all the JobIDs from the filtered dataframe
        user-ids: a list of all the UserIDs from the filtered dataframe
    '''
    df_apps_filtered = pd.read_hdf('stored-data/apps-filtered.h5', 'df')

    job_user_mat = df_apps_filtered.pivot(index='JobID', columns='UserID', values='Data').fillna(0)
    user_job_mat = job_user_mat.transpose()

    #create map from JobID to index in the job_user_mat
    job_ids = df_apps_filtered['JobID'].unique()
    filtered_jobs_df = pd.DataFrame(job_ids, columns=['JobID'])
    filtered_jobs_df['Extra_JobID'] = job_ids
    job_hashmap = {
        job: i for i, job in
        enumerate(list(filtered_jobs_df.set_index('JobID').loc[job_user_mat.index].Extra_JobID)) # noqa
    }

    #create map from UserID to index in the user_job_mat
    user_ids = df_apps_filtered['UserID'].unique()
    filtered_users_df = pd.DataFrame(user_ids, columns=['UserID'])
    filtered_users_df['Extra_UserID'] = user_ids
    user_hashmap = {
        user: i for i, user in
        enumerate(list(filtered_users_df.set_index('UserID').loc[user_job_mat.index].Extra_UserID)) # noqa
    }
    '''
    sparse_job_user = csr_matrix(job_user_mat.values) 
    save_npz('stored-data/sparse_job_user.npz', sparse_job_user)

    sparse_user_job = csr_matrix(user_job_mat.values)
    save_npz('stored-data/sparse_user_job.npz', sparse_user_job)
    
    print(sparse_user_job.shape) :
    With popularity_thres = 3    
         (17758, 19600)

    With popularity thres = 4
         (12415, 14778)
    
    With popularity thres = 5
         (9123, 11753)
    
    With popularity thres = 14
         (1510, 3367)
    '''

    return job_hashmap, user_hashmap, job_ids, user_ids


def job_and_user_info(job_ids, user_ids):
    '''
        Creates a datafame containing the job and user info in a form which can be used in ClearHire.
        Saves the dataframes as h5 files.
    '''
    job_info = pd.read_csv(wd + "jobs5.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="", engine='python')
    job_info = job_info.drop('StartDate', axis=1)
    job_info = job_info.drop('EndDate', axis=1)
    job_info = job_info.drop('WindowID', axis=1)
    job_info_filtered = job_info[job_info.JobID.isin(job_ids)]

    job_info_filtered['Description'] = job_info_filtered['Description'].map(lambda text: BeautifulSoup(text, 'html.parser').get_text().replace(r'\n',' ').replace(r'\r', ' ').replace(r'\t', ' '))
    job_info_filtered['Description'] = job_info_filtered['Description'].map(lambda text: " ".join(text.split()))

    job_info_filtered['Requirements'] = job_info_filtered['Requirements'].map(lambda text: text if isinstance(text, float) else BeautifulSoup(text, 'html.parser').get_text().replace(r'\n',' ').replace(r'\r', ' ').replace(r'\t', ' '))
    job_info_filtered['Requirements'] = job_info_filtered['Requirements'].map(lambda text: text if isinstance(text, float) else " ".join(text.split()))

    user_info = pd.read_csv(wd + "users5.tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="", engine='python')
    user_info = user_info.drop('WindowID', axis=1)
    user_info_filtered = user_info[user_info.UserID.isin(user_ids)]

    job_info_filtered.to_hdf('stored-data/job-info.h5', key='df', mode='w')
    user_info_filtered.to_hdf('stored-data/user-info.h5', key='df', mode='w')

    return job_info_filtered, user_info_filtered


def random_jobs(k=50):
    '''
    Returns k random jobs from the job-info dataframe
    '''
    job_info = pd.read_hdf('stored-data/job-info.h5', 'df')
    samples = job_info.sample(n = k)
    return samples



