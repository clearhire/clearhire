import pandas as pd
import numpy as np
import math
import dash_html_components as html
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors

def similar_users(user_job_mat, user_hashmap, job_hashmap, selected_jobs, n=4):
    '''
    Takes a list of the jobs selected by the ClearHire user. 
    Returns a list of the user-ids of the n most similar users.
    '''
    alpha = 40

    selected_job_ids = [row["JobID"] for _, row in selected_jobs.iterrows()]
    selected_job_indices = [job_hashmap[job_id] for job_id in selected_job_ids]

    #ensures user_job_mat is the correct shape
    user_job_mat = user_job_mat[:9123, :]
    n_users, n_jobs = user_job_mat.shape

    ratings = [alpha for i in range(len(selected_job_indices))]

    user_job_new_mat = user_job_mat
    user_job_new_mat.data = np.hstack((user_job_mat.data, ratings))
    user_job_new_mat.indices = np.hstack((user_job_mat.indices, selected_job_indices))
    user_job_new_mat.indptr = np.hstack((user_job_mat.indptr, len(user_job_mat.data)))
    user_job_new_mat._shape = (n_users+1, n_jobs)


    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_job_new_mat)

    distances, indices = model.kneighbors(user_job_new_mat[n_users], n_neighbors=n+1)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]) [:0:-1]

    reverse_mapper = {v: k for k, v in user_hashmap.items()}

    similar_users = []
    for _, (idx, _) in enumerate(raw_recommends):
        similar_users = similar_users + [reverse_mapper[idx]]

    return similar_users


def user_cf_recommend_jobs(user_hashmap, job_hashmap, selected_jobs, n=10):
    '''
    Returns a list of job ids for the top n recommendations, and the user_ids of the most similar users
    '''
    user_job_mat = load_npz('stored-data/sparse_user_job.npz')

    nearest_neighbours = similar_users(user_job_mat, user_hashmap, job_hashmap, selected_jobs)
    recommended_jobs_indices = []

    for i in nearest_neighbours:
        user_index = user_hashmap[i]
        user_row = user_job_mat.getrow(user_index)
        (_, nonzero_columns) = user_row.nonzero()
        recommended_jobs_indices = recommended_jobs_indices + nonzero_columns.tolist()

    reverse_job_mapper = {v: k for k, v in job_hashmap.items()}

    unique_recommendations = []

    for x in recommended_jobs_indices: 
        if x not in unique_recommendations: 
            unique_recommendations.append(reverse_job_mapper[x]) 
    
    return unique_recommendations[:n], nearest_neighbours


def user_information(user_ids):
    '''
    Takes the user-ids for the similar users, and returns the global explanation for the User-Global model.
    '''
    user_info = pd.read_hdf('stored-data/user-info.h5', 'df')

    user_degrees = pd.Series()  
    user_years_work_experience = pd.Series()
    user_number_managed = pd.Series()    
    
    for user_id in user_ids:
        user_row = user_info[user_info.UserID == user_id]

        user_years_work_experience = user_years_work_experience.append(user_row['TotalYearsExperience'])
        user_number_managed = user_number_managed.append(user_row['ManagedHowMany'])
        user_degrees = user_degrees.append(user_row['DegreeType'])

    #to format the string    
    average_degree = user_degrees.mode().to_string()
    average_degree_remove_int = average_degree.split(' ', 1)[1]
    average_degree_formatted = average_degree_remove_int.split()[0]
    if (average_degree_formatted == "None"):
        average_degree_formatted = "no"
    elif (average_degree_formatted == "High"):
        average_degree_formatted = "a High School"
    else:
         average_degree_formatted = "a " + average_degree_formatted

    average_years_experience = user_years_work_experience.mean()
    if not math.isnan(average_years_experience):
        average_years_experience = int(average_years_experience)

    average_number_managed = user_number_managed.mean()
    if not math.isnan(average_number_managed):
        average_number_managed = int(average_number_managed)

    description = ['The algorithm first searched for users who applied to jobs that are similar to the ones you selected, and then generated recommendations based on other jobs these users have applied for. You are similar to users 1, 2 and 3. They have average qualifications of:',
                    html.Br(),
                    html.Ul(children=[
                        html.Li('{} degree'.format(average_degree_formatted)),
                        html.Li('{} years previous work experience'.format(average_years_experience)),
                        html.Li('management experience of > {} other people'.format(average_number_managed) )
                    ])]    

    return description



def user_cf_map_jobs(jobIDs):
    '''
    Returns a dataframe containing all the job information to be shown in the User-Global model in ClearHire for the jobs in the jobIDs.    
    '''
    job_info = pd.read_hdf('stored-data/job-info.h5', 'df')

    job_description = job_info[job_info.JobID.isin(jobIDs)]
    return job_description
