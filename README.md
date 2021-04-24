# ClearHire

ClearHire is a platform to experiment with different explanations and algorithms for job listing sites. 
It was designed to as part of an online experiment to evaluate how different styles of explanation can help users to understand 
the possible biases which may be present in such sites.

Study participants had to pretend they were searching for a jobs. Upon loading the site, a list of jobs was presented to them from which they
were asked to select at least three they would be interested in applying to. Four different lists of recommendations were then generated: Option 1 and 
Option 2 use the same matrix factorisation algorithm, Option 3 uses a user-based collaborative filtering algorithm, and Option 4 uses an item-based 
collaborative filtering algorithm. Each Option has their own unique explanation style.

This repository contains the following files:
- __data_manipulation.py__ contains the functions used to process the dataset and create the user-job matrices used by all the algorithms.
- __mf_model.py__ contains the code for the matrix factorisation algorithm, as well as the explanation for Option 1.
- __user_cf.py__ contains the code for the user-based collaborative filtering algorithm and explanation.
- __item_cf.py__ contains the code for the item-based collaborative filtering algorithm and explanation.
- __database_explanation.py__ contains the code for the explanation for Option 2.
- __selection_table.py__ is used to generate the list of jobs presented to users when they first load the site.
- __app.py__ contains the code for the design and functionality the website. This is done using Plotly Dash.
- __website_helper.py__ contains functions to help generating various features for the website.
- __hyperparameter_tuning.py__ uses a validation set to tune the hyperparameters for the matrix factorisation model.
- __accuracy.py__ compares the accuracy of the three algorithms on a test set.
- __data_analysis.py__ was used to analyse the original dataset.

It should be noted that the original dataset is not uploaded in this repository, but it can be found at:
https://www.kaggle.com/c/job-recommendation/data.
However, all the necessary data to run ClearHire is preprocessed and saved in the stored-data folder.

## General Usage Notes ##
The link to ClearHire is: https://clearhire.herokuapp.com.

It should be accessed on a computer for a proper display. Please note that the dynos used to host this application go into sleep mode if not used for thirty seconds,
so the website may take a few seconds to load as the dynos need time to start

