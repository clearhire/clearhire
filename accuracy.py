'''
Calculates the accuracy of the three algorithms on the test dataset.
'''
import implicit
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from cross_validation import recall, at_least_one_metric

def calculate_mf_recommendations(mf_model, sparse_user_job, reduced_sparse_user_job, user_index, n=10):
    '''
    Input:
        user_index: the index of the user to calculate the recommendations for in the user-job matrix
    Adds the first three applications made by the user as a new user to sparse matrix and generates recommendations with the ALS algorithm. 
    Returns the recall and atLeastOne accuracy metrics calculated based on the remaining applications made by this user.
    '''

    alpha = 40

    row = sparse_user_job[np.array([user_index]),:]
    (_, nonzero_columns) = row.nonzero()
    training_columns = nonzero_columns[:3]
    test_columns = nonzero_columns[3:]

    #ensures user_job_mat is the correct shape
    reduced_sparse_user_job = reduced_sparse_user_job[:7812, :]
    n_users, n_jobs = reduced_sparse_user_job.shape

    train_ratings = [alpha for i in range(3)]

    new_sparse_user_job = reduced_sparse_user_job
    new_sparse_user_job.data = np.hstack((reduced_sparse_user_job.data, train_ratings))
    new_sparse_user_job.indices = np.hstack((reduced_sparse_user_job.indices, training_columns))
    new_sparse_user_job.indptr = np.hstack((reduced_sparse_user_job.indptr, len(reduced_sparse_user_job.data)))
    new_sparse_user_job._shape = (n_users+1, n_jobs)

    recommended_index, _ =  zip(*mf_model.recommend(n_users, new_sparse_user_job, N=n, recalculate_user=True))
    
    return recall(recommended_index, test_columns), at_least_one_metric(recommended_index, test_columns)


def mf_testing(sparse_user_job):
    '''
    Calculates the recall and atLeastOne metric of the ALS algorithm on the test dataset.
    '''

    sparse_job_user = sparse_user_job.T

    columns = np.arange(0,7812)
    sparse_job_user_reduced = sparse_job_user[:, columns]        
    sparse_user_job_reduced = sparse_job_user_reduced.T

    mf_model = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    mf_model.fit(sparse_job_user_reduced)

    recall_total = 0
    at_least_one_metric_total = 0
    for i in range(7812, 9123):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model, sparse_user_job, sparse_user_job_reduced, i)
        recall_total = recall_total + recall
        at_least_one_metric_total = at_least_one_metric_total + at_least_one_metric
    
    average_recall = float(recall_total) / 1311
    average_at_least_one_metric = float(at_least_one_metric_total) / 1311
    print('average_recall: ', average_recall)
    print('average_at_least_one_metric: ', average_at_least_one_metric)

    '''
        average_recall:  0.3000408253533875
        average_at_least_one_metric:  0.7841342486651411
    '''


def calculate_user_cf_recommendations(sparse_user_job, reduced_sparse_user_job, user_index, n=10):
    ''' 
    Input:
        user_index: the index of the user to calculate the recommendations for in the user-job matrix
    Adds the first three applications made by the user as a new user to sparse matrix and generates recommendations with the user-cf algorithm. 
    Returns the recall and atLeastOne accuracy metrics calculated based on the remaining applications made by this user.
    '''
    alpha = 40

    row = sparse_user_job[np.array([user_index]),:]
    (_, nonzero_columns) = row.nonzero()
    training_columns = nonzero_columns[:3]
    test_columns = nonzero_columns[3:]

    #ensures user_job_mat is the correct shape
    reduced_sparse_user_job = reduced_sparse_user_job[:7812, :]
    n_users, n_jobs = reduced_sparse_user_job.shape

    train_ratings = [alpha for i in range(3)]

    user_job_new_mat = reduced_sparse_user_job
    user_job_new_mat.data = np.hstack((reduced_sparse_user_job.data, train_ratings))
    user_job_new_mat.indices = np.hstack((reduced_sparse_user_job.indices, training_columns))
    user_job_new_mat.indptr = np.hstack((reduced_sparse_user_job.indptr, len(reduced_sparse_user_job.data)))
    user_job_new_mat._shape = (n_users+1, n_jobs)


    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_job_new_mat)

    distances, indices = model.kneighbors(user_job_new_mat[n_users], n_neighbors=n+1)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]) [:0:-1]

    recommended_jobs_indices = []
    for _, (idx, _) in enumerate(raw_recommends):
        user_row = reduced_sparse_user_job.getrow(idx)
        (_, nonzero_columns) = user_row.nonzero()
        recommended_jobs_indices = recommended_jobs_indices + nonzero_columns.tolist()

    unique_recommendations = []

    for x in recommended_jobs_indices: 
        if x not in unique_recommendations: 
            unique_recommendations.append(x) 

    final_recommendations = unique_recommendations[:n]
    
    return recall(final_recommendations, test_columns), at_least_one_metric(final_recommendations, test_columns)


def user_cf_testing(sparse_user_job):
    '''
    Calculates the recall and atLeastOne metric of the user-cf algorithm on the test dataset.
    '''
    rows = np.arange(0,7812)
    sparse_user_job_reduced = sparse_user_job[rows, :]        

    recall_total = 0
    at_least_one_metric_total = 0
    for i in range(7812, 9123):
        recall, at_least_one_metric = calculate_user_cf_recommendations(sparse_user_job, sparse_user_job_reduced, i)
        recall_total = recall_total + recall
        at_least_one_metric_total = at_least_one_metric_total + at_least_one_metric
    
    average_recall = float(recall_total) / 1311
    average_at_least_one_metric = float(at_least_one_metric_total) / 1311
    print('average_recall: ', average_recall)
    print('average_at_least_one_metric: ', average_at_least_one_metric)

    '''
        average_recall:  0.08709592698637737
        average_at_least_one_metric:  0.38443935926773454
    '''


def calculate_job_cf_recommendations(sparse_job_user, reduced_sparse_job_user, user_index, n=10):
    ''' 
    Input:
        user_index: the index of the user to calculate the recommendations for in the user-job matrix
    Adds the first three applications made by the user as a new user to sparse matrix and generates recommendations with the job-cf algorithm. 
    Returns the recall and atLeastOne accuracy metrics calculated based on the remaining applications made by this user.
    '''
    column = sparse_job_user[:, np.array([user_index])]
    (nonzero_rows, _) = column.nonzero()
    training_rows = nonzero_rows[:3]
    test_rows = nonzero_rows[3:]

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(reduced_sparse_job_user)

    #list of the lists of recommendations from each job in selected_jobs
    possible_recommendations = []
    for job_index in training_rows:
        distances, indices = model.kneighbors(reduced_sparse_job_user[job_index], n_neighbors=n+1)
        raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1]) [:0:-1]

        similar_jobs = []
        for _, (idx, _) in enumerate(raw_recommends):
            similar_jobs = similar_jobs + [idx]
        
        possible_recommendations = possible_recommendations + [(job_index, similar_jobs)]
    
    unique_recommendations = []
    for i in range(10):
        for (job_index, job_list) in possible_recommendations:
            job = job_list[i]
            if job not in unique_recommendations: 
                unique_recommendations.append(job)
    
    
    final_recommendations = unique_recommendations[:n]
    
    return recall(final_recommendations, test_rows), at_least_one_metric(final_recommendations, test_rows)


def job_cf_testing(sparse_job_user):
    '''
    Calculates the recall and atLeastOne metric of the job-cf algorithm on the test dataset.
    '''
    columns = np.arange(0,7812)
    sparse_job_user_reduced = sparse_job_user[:, columns]        

    recall_total = 0
    at_least_one_metric_total = 0
    for i in range(7812, 9123):
        recall, at_least_one_metric = calculate_job_cf_recommendations(sparse_job_user, sparse_job_user_reduced, i)
        recall_total = recall_total + recall
        at_least_one_metric_total = at_least_one_metric_total + at_least_one_metric
    
    average_recall = float(recall_total) / 1311
    average_at_least_one_metric = float(at_least_one_metric_total) / 1311
    print('average_recall: ', average_recall)
    print('average_at_least_one_metric: ', average_at_least_one_metric)
    
    '''
        average_recall:  0.14926222177997966
        average_at_least_one_metric:  0.5789473684210527 
    '''
