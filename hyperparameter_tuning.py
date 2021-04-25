'''
Split users into 14 groups: 11 for training, 1 for validation, and 2 for testing.
Each group has 651 users.
The code in accuracy.py was used for testing, while the code in this file was used for hyperparameter tuning of the ALS algorithm:
Each of the 12 possible models were trained on the training data and then tested on the validation set. 
For each of the test users suppose they have selected 3 jobs and calculate recall + was at least one recommended for remaining applications.
The 12 models tested were:  1 - fac=10, reg=0.3
                            2 - fac=10, reg=0.1
                            3 - fac=10, reg=0.01
                            4 - fac=50, reg=0.3
                            5 - fac=50, reg=0.1
                            6 - fac=50, reg=0.01
                            7 - fac=100, reg=0.3
                            8 - fac=100, reg=0.1
                            9 - fac=100, reg=0.01
                            10 - fac=200, reg=0.3
                            11 - fac=200, reg=0.1
                            12 - fac=200, reg=0.01
                
We found the optimum hyperparameter setting to be: 
 implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
'''

import implicit
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def recall(recommended_jobs, test_jobs):
    number_test = len(test_jobs)
    number_test_recommended = 0
    for job in test_jobs:
        if job in recommended_jobs:
            number_test_recommended = number_test_recommended + 1
    
    return ( float(number_test_recommended) / float(number_test) )


def at_least_one_metric(recommended_jobs, test_jobs):
    '''
    Takes a list of recommended job ids and a list of test jobs ids. 
    Returns whether at least one of the test job ids have been recommended.
    '''

    success = 0
    for job in test_jobs:
        if job in recommended_jobs:
            success = 1

    return success


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
    reduced_sparse_user_job = reduced_sparse_user_job[:7161, :]
    n_users, n_jobs = reduced_sparse_user_job.shape

    train_ratings = [alpha for i in range(3)]

    new_sparse_user_job = reduced_sparse_user_job
    new_sparse_user_job.data = np.hstack((reduced_sparse_user_job.data, train_ratings))
    new_sparse_user_job.indices = np.hstack((reduced_sparse_user_job.indices, training_columns))
    new_sparse_user_job.indptr = np.hstack((reduced_sparse_user_job.indptr, len(reduced_sparse_user_job.data)))
    new_sparse_user_job._shape = (n_users+1, n_jobs)

    recommended_index, _ =  zip(*mf_model.recommend(n_users, new_sparse_user_job, N=n, recalculate_user=True))
    
    return recall(recommended_index, test_columns), at_least_one_metric(recommended_index, test_columns)


def hyperparameter_tuning(sparse_user_job):
    '''
    Original calculations of the hyperparamter setting which yielded the highest accuracy on the validation set.
    '''
    
    sparse_job_user = sparse_user_job.T

    columns = np.arange(651,7812)
    sparse_job_user_reduced = sparse_job_user[:, columns]        
    sparse_user_job_reduced = sparse_job_user_reduced.T

    mf_model_1 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.3, iterations=30)
    mf_model_1.fit(sparse_job_user_reduced)

    mf_model_2 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1, iterations=30)
    mf_model_2.fit(sparse_job_user_reduced)

    mf_model_3 = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.01, iterations=30)
    mf_model_3.fit(sparse_job_user_reduced)

    mf_model_4 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.3, iterations=30)
    mf_model_4.fit(sparse_job_user_reduced)

    mf_model_5 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    mf_model_5.fit(sparse_job_user_reduced)

    mf_model_6 = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=30)
    mf_model_6.fit(sparse_job_user_reduced)

    mf_model_7 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.3, iterations=30)
    mf_model_7.fit(sparse_job_user_reduced)

    mf_model_8 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.1, iterations=30)
    mf_model_8.fit(sparse_job_user_reduced)

    mf_model_9 = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.01, iterations=30)
    mf_model_9.fit(sparse_job_user_reduced)

    mf_model_10 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.3, iterations=30)
    mf_model_10.fit(sparse_job_user_reduced)

    mf_model_11 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=30)
    mf_model_11.fit(sparse_job_user_reduced)

    mf_model_12 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    mf_model_12.fit(sparse_job_user_reduced)


    recall_1 = 0
    at_least_one_metric_1 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_1, sparse_user_job, sparse_user_job_reduced, i)
        recall_1 = recall_1 + recall
        at_least_one_metric_1 = at_least_one_metric_1 + at_least_one_metric
    
    average_recall_1 = float(recall_1) / 651
    average_at_least_one_metric_1 = float(at_least_one_metric_1) / 651
    print('average_recall_1: ', average_recall_1)
    print('average_at_least_one_metric_1: ', average_at_least_one_metric_1)

    
    recall_2 = 0
    at_least_one_metric_2 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_2, sparse_user_job, sparse_user_job_reduced, i)
        recall_2 = recall_2 + recall
        at_least_one_metric_2 = at_least_one_metric_2 + at_least_one_metric
    
    average_recall_2 = float(recall_2) / 651
    average_at_least_one_metric_2 = float(at_least_one_metric_2) / 651
    print('average_recall_2: ', average_recall_2)
    print('average_at_least_one_metric_2: ', average_at_least_one_metric_2)

    recall_3 = 0
    at_least_one_metric_3 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_3, sparse_user_job, sparse_user_job_reduced, i)
        recall_3 = recall_3 + recall
        at_least_one_metric_3 = at_least_one_metric_3 + at_least_one_metric
    
    average_recall_3 = float(recall_3) / 651
    average_at_least_one_metric_3 = float(at_least_one_metric_3) / 651
    print('average_recall_3: ', average_recall_3)
    print('average_at_least_one_metric_3: ', average_at_least_one_metric_3)

    recall_4 = 0
    at_least_one_metric_4 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_4, sparse_user_job, sparse_user_job_reduced, i)
        recall_4 = recall_4 + recall
        at_least_one_metric_4 = at_least_one_metric_4 + at_least_one_metric
    
    average_recall_4 = float(recall_4) / 651
    average_at_least_one_metric_4 = float(at_least_one_metric_4) / 651
    print('average_recall_4: ', average_recall_4)
    print('average_at_least_one_metric_4: ', average_at_least_one_metric_4)

    recall_5 = 0
    at_least_one_metric_5 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_5, sparse_user_job, sparse_user_job_reduced, i)
        recall_5 = recall_5 + recall
        at_least_one_metric_5 = at_least_one_metric_5 + at_least_one_metric
    
    average_recall_5 = float(recall_5) / 651
    average_at_least_one_metric_5 = float(at_least_one_metric_5) / 651
    print('average_recall_5: ', average_recall_5)
    print('average_at_least_one_metric_5: ', average_at_least_one_metric_5)

    recall_6 = 0
    at_least_one_metric_6 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_6, sparse_user_job, sparse_user_job_reduced, i)
        recall_6 = recall_6 + recall
        at_least_one_metric_6 = at_least_one_metric_6 + at_least_one_metric
    
    average_recall_6 = float(recall_6) / 651
    average_at_least_one_metric_6 = float(at_least_one_metric_6) / 651
    print('average_recall_6: ', average_recall_6)
    print('average_at_least_one_metric_6: ', average_at_least_one_metric_6)

    recall_7 = 0
    at_least_one_metric_7 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_7, sparse_user_job, sparse_user_job_reduced, i)
        recall_7 = recall_7 + recall
        at_least_one_metric_7 = at_least_one_metric_7 + at_least_one_metric
    
    average_recall_7 = float(recall_7) / 651
    average_at_least_one_metric_7 = float(at_least_one_metric_7) / 651
    print('average_recall_7: ', average_recall_7)
    print('average_at_least_one_metric_7: ', average_at_least_one_metric_7)

    recall_8 = 0
    at_least_one_metric_8 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_8, sparse_user_job, sparse_user_job_reduced, i)
        recall_8 = recall_8 + recall
        at_least_one_metric_8 = at_least_one_metric_8 + at_least_one_metric
    
    average_recall_8 = float(recall_8) / 651
    average_at_least_one_metric_8 = float(at_least_one_metric_8) / 651
    print('average_recall_8: ', average_recall_8)
    print('average_at_least_one_metric_8: ', average_at_least_one_metric_8)

    recall_9 = 0
    at_least_one_metric_9 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_9, sparse_user_job, sparse_user_job_reduced, i)
        recall_9 = recall_9 + recall
        at_least_one_metric_9 = at_least_one_metric_9 + at_least_one_metric
    
    average_recall_9 = float(recall_9) / 651
    average_at_least_one_metric_9 = float(at_least_one_metric_9) / 651
    print('average_recall_9: ', average_recall_9)
    print('average_at_least_one_metric_9: ', average_at_least_one_metric_9)

    recall_10 = 0
    at_least_one_metric_10 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_10, sparse_user_job, sparse_user_job_reduced, i)
        recall_10 = recall_10 + recall
        at_least_one_metric_10 = at_least_one_metric_10 + at_least_one_metric
    
    average_recall_10 = float(recall_10) / 651
    average_at_least_one_metric_10 = float(at_least_one_metric_10) / 651
    print('average_recall_10: ', average_recall_10)
    print('average_at_least_one_metric_10: ', average_at_least_one_metric_10)

    recall_11 = 0
    at_least_one_metric_11 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_11, sparse_user_job, sparse_user_job_reduced, i)
        recall_11 = recall_11 + recall
        at_least_one_metric_11 = at_least_one_metric_11 + at_least_one_metric
    
    average_recall_11 = float(recall_11) / 651
    average_at_least_one_metric_11 = float(at_least_one_metric_11) / 651
    print('average_recall_11: ', average_recall_11)
    print('average_at_least_one_metric_11: ', average_at_least_one_metric_11)

    recall_12 = 0
    at_least_one_metric_12 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_12, sparse_user_job, sparse_user_job_reduced, i)
        recall_12 = recall_12 + recall
        at_least_one_metric_12 = at_least_one_metric_12 + at_least_one_metric
    
    average_recall_12 = float(recall_12) / 651
    average_at_least_one_metric_12 = float(at_least_one_metric_12) / 651
    print('average_recall_12: ', average_recall_12)
    print('average_at_least_one_metric_12: ', average_at_least_one_metric_12)
    
    '''
        average_recall_1:  0.10671541126115974
        average_at_least_one_metric_1:  0.41321044546851

        average_recall_2:  0.09912884792161074
        average_at_least_one_metric_2:  0.3655913978494624

        average_recall_3:  0.11492882635904957
        average_at_least_one_metric_3:  0.4116743471582181

        average_recall_4:  0.25428424194053845
        average_at_least_one_metric_4:  0.7096774193548387

        average_recall_5:  0.26017080120096564
        average_at_least_one_metric_5:  0.738863287250384

        average_recall_6:  0.25657350392394596
        average_at_least_one_metric_6:  0.706605222734255

        average_recall_7:  0.26967534593747905
        average_at_least_one_metric_7:  0.7542242703533026

        average_recall_8:  0.2868122891005407
        average_at_least_one_metric_8:  0.7542242703533026

        average_recall_9:  0.26453803897856293
        average_at_least_one_metric_9:  0.7603686635944701

        average_recall_10:  0.29583776898816283
        average_at_least_one_metric_10:  0.8018433179723502

        average_recall_11:  0.2874163814756907
        average_at_least_one_metric_11:  0.7695852534562212

        average_recall_12:  0.296975774517057
        average_at_least_one_metric_12:  0.7910906298003072
    '''

def further_hyperparameter_tuning(sparse_user_job):
    '''
    Run after having run the hyperparamter_tuning function above. 
    This further narrowed down which hyperparameters yielded the highest accuracy on the validation set.
    '''
    sparse_job_user = sparse_user_job.T

    columns = np.arange(651,7812)
    sparse_job_user_reduced = sparse_job_user[:, columns]        
    sparse_user_job_reduced = sparse_job_user_reduced.T

    mf_model_1 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.3, iterations=30)
    mf_model_1.fit(sparse_job_user_reduced)

    mf_model_2 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.1, iterations=30)
    mf_model_2.fit(sparse_job_user_reduced)

    mf_model_3 = implicit.als.AlternatingLeastSquares(factors=150, regularization=0.01, iterations=30)
    mf_model_3.fit(sparse_job_user_reduced)

    mf_model_4 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.3, iterations=30)
    mf_model_4.fit(sparse_job_user_reduced)

    mf_model_5 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.1, iterations=30)
    mf_model_5.fit(sparse_job_user_reduced)

    mf_model_6 = implicit.als.AlternatingLeastSquares(factors=200, regularization=0.01, iterations=30)
    mf_model_6.fit(sparse_job_user_reduced)

    mf_model_7 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.3, iterations=30)
    mf_model_7.fit(sparse_job_user_reduced)

    mf_model_8 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.1, iterations=30)
    mf_model_8.fit(sparse_job_user_reduced)

    mf_model_9 = implicit.als.AlternatingLeastSquares(factors=250, regularization=0.01, iterations=30)
    mf_model_9.fit(sparse_job_user_reduced)

    mf_model_10 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.3, iterations=30)
    mf_model_10.fit(sparse_job_user_reduced)

    mf_model_11 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.1, iterations=30)
    mf_model_11.fit(sparse_job_user_reduced)

    mf_model_12 = implicit.als.AlternatingLeastSquares(factors=300, regularization=0.01, iterations=30)
    mf_model_12.fit(sparse_job_user_reduced)


    recall_1 = 0
    at_least_one_metric_1 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_1, sparse_user_job, sparse_user_job_reduced, i)
        recall_1 = recall_1 + recall
        at_least_one_metric_1 = at_least_one_metric_1 + at_least_one_metric
    
    average_recall_1 = float(recall_1) / 651
    average_at_least_one_metric_1 = float(at_least_one_metric_1) / 651
    print('average_recall_1: ', average_recall_1)
    print('average_at_least_one_metric_1: ', average_at_least_one_metric_1)

    
    recall_2 = 0
    at_least_one_metric_2 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_2, sparse_user_job, sparse_user_job_reduced, i)
        recall_2 = recall_2 + recall
        at_least_one_metric_2 = at_least_one_metric_2 + at_least_one_metric
    
    average_recall_2 = float(recall_2) / 651
    average_at_least_one_metric_2 = float(at_least_one_metric_2) / 651
    print('average_recall_2: ', average_recall_2)
    print('average_at_least_one_metric_2: ', average_at_least_one_metric_2)

    recall_3 = 0
    at_least_one_metric_3 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_3, sparse_user_job, sparse_user_job_reduced, i)
        recall_3 = recall_3 + recall
        at_least_one_metric_3 = at_least_one_metric_3 + at_least_one_metric
    
    average_recall_3 = float(recall_3) / 651
    average_at_least_one_metric_3 = float(at_least_one_metric_3) / 651
    print('average_recall_3: ', average_recall_3)
    print('average_at_least_one_metric_3: ', average_at_least_one_metric_3)

    recall_4 = 0
    at_least_one_metric_4 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_4, sparse_user_job, sparse_user_job_reduced, i)
        recall_4 = recall_4 + recall
        at_least_one_metric_4 = at_least_one_metric_4 + at_least_one_metric
    
    average_recall_4 = float(recall_4) / 651
    average_at_least_one_metric_4 = float(at_least_one_metric_4) / 651
    print('average_recall_4: ', average_recall_4)
    print('average_at_least_one_metric_4: ', average_at_least_one_metric_4)

    recall_5 = 0
    at_least_one_metric_5 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_5, sparse_user_job, sparse_user_job_reduced, i)
        recall_5 = recall_5 + recall
        at_least_one_metric_5 = at_least_one_metric_5 + at_least_one_metric
    
    average_recall_5 = float(recall_5) / 651
    average_at_least_one_metric_5 = float(at_least_one_metric_5) / 651
    print('average_recall_5: ', average_recall_5)
    print('average_at_least_one_metric_5: ', average_at_least_one_metric_5)

    recall_6 = 0
    at_least_one_metric_6 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_6, sparse_user_job, sparse_user_job_reduced, i)
        recall_6 = recall_6 + recall
        at_least_one_metric_6 = at_least_one_metric_6 + at_least_one_metric
    
    average_recall_6 = float(recall_6) / 651
    average_at_least_one_metric_6 = float(at_least_one_metric_6) / 651
    print('average_recall_6: ', average_recall_6)
    print('average_at_least_one_metric_6: ', average_at_least_one_metric_6)

    recall_7 = 0
    at_least_one_metric_7 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_7, sparse_user_job, sparse_user_job_reduced, i)
        recall_7 = recall_7 + recall
        at_least_one_metric_7 = at_least_one_metric_7 + at_least_one_metric
    
    average_recall_7 = float(recall_7) / 651
    average_at_least_one_metric_7 = float(at_least_one_metric_7) / 651
    print('average_recall_7: ', average_recall_7)
    print('average_at_least_one_metric_7: ', average_at_least_one_metric_7)

    recall_8 = 0
    at_least_one_metric_8 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_8, sparse_user_job, sparse_user_job_reduced, i)
        recall_8 = recall_8 + recall
        at_least_one_metric_8 = at_least_one_metric_8 + at_least_one_metric
    
    average_recall_8 = float(recall_8) / 651
    average_at_least_one_metric_8 = float(at_least_one_metric_8) / 651
    print('average_recall_8: ', average_recall_8)
    print('average_at_least_one_metric_8: ', average_at_least_one_metric_8)

    recall_9 = 0
    at_least_one_metric_9 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_9, sparse_user_job, sparse_user_job_reduced, i)
        recall_9 = recall_9 + recall
        at_least_one_metric_9 = at_least_one_metric_9 + at_least_one_metric
    
    average_recall_9 = float(recall_9) / 651
    average_at_least_one_metric_9 = float(at_least_one_metric_9) / 651
    print('average_recall_9: ', average_recall_9)
    print('average_at_least_one_metric_9: ', average_at_least_one_metric_9)

    recall_10 = 0
    at_least_one_metric_10 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_10, sparse_user_job, sparse_user_job_reduced, i)
        recall_10 = recall_10 + recall
        at_least_one_metric_10 = at_least_one_metric_10 + at_least_one_metric
    
    average_recall_10 = float(recall_10) / 651
    average_at_least_one_metric_10 = float(at_least_one_metric_10) / 651
    print('average_recall_10: ', average_recall_10)
    print('average_at_least_one_metric_10: ', average_at_least_one_metric_10)

    recall_11 = 0
    at_least_one_metric_11 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_11, sparse_user_job, sparse_user_job_reduced, i)
        recall_11 = recall_11 + recall
        at_least_one_metric_11 = at_least_one_metric_11 + at_least_one_metric
    
    average_recall_11 = float(recall_11) / 651
    average_at_least_one_metric_11 = float(at_least_one_metric_11) / 651
    print('average_recall_11: ', average_recall_11)
    print('average_at_least_one_metric_11: ', average_at_least_one_metric_11)

    recall_12 = 0
    at_least_one_metric_12 = 0
    for i in range(0, 651):
        recall, at_least_one_metric = calculate_mf_recommendations(mf_model_12, sparse_user_job, sparse_user_job_reduced, i)
        recall_12 = recall_12 + recall
        at_least_one_metric_12 = at_least_one_metric_12 + at_least_one_metric
    
    average_recall_12 = float(recall_12) / 651
    average_at_least_one_metric_12 = float(at_least_one_metric_12) / 651
    print('average_recall_12: ', average_recall_12)
    print('average_at_least_one_metric_12: ', average_at_least_one_metric_12)

    '''
        average_recall_1:  0.2984770737307024
        average_at_least_one_metric_1:  0.7803379416282642

        average_recall_2:  0.2820620919866327
        average_at_least_one_metric_2:  0.7526881720430108

        average_recall_3:  0.2790073031455297
        average_at_least_one_metric_3:  0.7511520737327189

        average_recall_4:  0.30420545254210585
        average_at_least_one_metric_4:  0.7772657450076805

        average_recall_5:  0.31866686100323754
        average_at_least_one_metric_5:  0.8095238095238095

        average_recall_6:  0.3135713776516499
        average_at_least_one_metric_6:  0.804915514592934

        average_recall_7:  0.30271406916576993
        average_at_least_one_metric_7:  0.8095238095238095

        average_recall_8:  0.3052751653310733
        average_at_least_one_metric_8:  0.7803379416282642

        average_recall_9:  0.2916355589588203
        average_at_least_one_metric_9:  0.7926267281105991

        average_recall_10:  0.30363246667898364
        average_at_least_one_metric_10:  0.8110599078341014

        average_recall_11:  0.2985783391877972
        average_at_least_one_metric_11:  0.7895545314900153

        average_recall_12:  0.29369413847955506
        average_at_least_one_metric_12:  0.7818740399385561
    '''

