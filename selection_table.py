'''
creates the list of jobs from which ClearHire users can select the ones that interest them most.
'''

import pandas as pd

job_ids_1 = [886760, 63406, 1082395, 564500, 575179, 584172]
job_ids_2 = [171824, 347307, 959967, 254753, 850644]
job_ids_3 = [880174, 81716, 392712, 600723, 849522, 546284]
job_ids_4 = [630368, 361497, 508742, 646987, 872334, 1041557, 6386]
job_ids_5 = [525424, 310800, 877829, 78799, 132835]
job_ids_6 = [283201, 581983, 582110, 28626, 452709, 799012]

job_info = pd.read_hdf('stored-data/job-info.h5', 'df')
job_info_1 = job_info[job_info.JobID.isin(job_ids_1)]
job_info_2 = job_info[job_info.JobID.isin(job_ids_2)]
job_info_3 = job_info[job_info.JobID.isin(job_ids_3)]
job_info_4 = job_info[job_info.JobID.isin(job_ids_4)]
job_info_5 = job_info[job_info.JobID.isin(job_ids_5)]
job_info_6 = job_info[job_info.JobID.isin(job_ids_6)]

frames = [job_info_1, job_info_2, job_info_3, job_info_4, job_info_5, job_info_6]
selection_jobs = pd.concat(frames)

job_categories = ['Business and Management']*6 + ['Accountancy and Finance']*5 + ['Professional']*6 + ['Skilled Labour']*7 + ['Sales']*5 + ['Administration']*6
selection_jobs['Category'] = job_categories

selection_jobs.to_hdf('stored-data/selection_jobs.h5', key='df', mode='w')

