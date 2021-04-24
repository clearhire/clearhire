import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
import json
import base64
from data_manipulation import load_data, random_jobs
from user_cf import user_cf_recommend_jobs, user_information, user_cf_map_jobs
from item_cf import job_cf_recommend_jobs, item_cf_map_jobs
from mf_model import mf_recommend_jobs, mf_map_jobs
from website_helper import generate_table, generate_table_without_explanations, tooltip_data, generate_comparison_table
from database_explanation import db_explanation_map_jobs



job_hashmap, user_hashmap, job_ids, user_ids = load_data()
sample_jobs = pd.read_hdf('stored-data/selection_jobs.h5', 'df')

logo = 'stored-data/logo.png'
encoded_image = base64.b64encode(open(logo, 'rb').read())

external_stylesheets = ['https:codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(style={'backgroundColor': '#3c73a8'}, children=[
    html.Div(
        id='header', 
        children=[
            html.Div(
                children=[
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    height=200)
                ], 
                style={
                    'margin': '0px 0px 0px 465px'
                }
            ),
            html.Div(
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'fontSize': 70,
                }, 
                children='''ClearHire'''
            ),
            html.Div(
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'fontSize': 20,
                    'font-family': 'Open Sans',
                },
                children='A platform to experiment with different styles of explanations and algorithms for job listing sites',
            ),
        ], 
        style={'padding': 50}
    ),

    html.Div(
        style={
            'margin': '50px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 15,
        }, 
        children=['You want to apply for a new job. ClearHire allows you to compare different algorithms and explanations for job recommendations to find the one that suits you best.']
    ),
    html.Div(
        style={
            'margin': '25px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 15,
        }, 
        children=['Step 1: Carefully select at least three jobs from the list below that you would be interested in applying to. They have been divided into categories to help you choose.'],
    ),
    html.Div(
        style={
            'margin': '5px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 15,
        }, 
        children=['Step 2: Click the SUBMIT button which can be found at the bottom of the page.']
    ),
    html.Div(
        style={
            'margin': '5px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 15,
        }, 
        children=['Step 3: Scroll down to see four different lists of job recommendations, based on the jobs you selected.']
    ),
     html.Div(
        style={
            'margin': '25px 22px 0px 25px',
            'color': '#FFFFFF',
            'fontSize': 15,
        }, 
        children=['Hover your mouse over the text in the Description and Requirements columns to view more information about each job.']
    ),
    
    dash_table.DataTable(
        id='sample-jobs',
        columns=[
            {"name": 'Title', "id": 'Title'}, 
            {"name": 'Category', 'id': 'Category'},
            {"name": 'Description', "id": 'Description'}, 
            {"name": 'Requirements', "id": 'Requirements'}, 
            {"name": 'City', "id": 'City'}, 
            {"name": 'State', "id": 'State'}, 
            {"name": 'Country', "id": 'Country'}
        ], 
        data=sample_jobs.to_dict('records'),
        row_selectable='multi',
        selected_rows=[],
        style_cell={
            'textAlign': 'left',
            'font-family': 'Open Sans',
            'fontSize': 13,
            'backgroundColor': '#d9ecf3',
            'padding': '5px 10px 0px 12px', 
            'textOverflow': 'clip',
            'overflow': 'hidden',
            'maxWidth': '75px', 
            'minWidth': '75px',
            'width': '75px',
            'minHeight': '45px', 
            'maxHeight': '45px', 
            'height': '45px', 
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'Title'},
                'minWidth': '250px',
                'maxWidth': '250px',
                'width': '250px',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Requirements'},
                'minWidth': '245px',
                'maxWidth': '245px',
                'width': '245px',
            },
            {
                'if': {'column_id': 'Description'},
                'minWidth': '245px',
                'maxWidth': '245px',
                'width': '245px',
            },
            {
                'if': {'column_id': 'Category'},
                'minWidth': '170px',
                'maxWidth': '170px',
                'width': '170px',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'City'},
                'minWidth': '92px',
                'maxWidth': '92px',
                'width': '92px',
            },
        ],
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#a7dff3e8',
            'fontSize': 16,
            'minHeight': '60px',
            'height': '60px'
        },
        style_as_list_view=True,
        style_table={
            'border': '22px solid #3c73a8',
            'borderRadius': '15px',
            'width': '96%'
        },
        tooltip_data=[
            {
                column: {'value': str(value)}
                for column, value in row.items()
            } for row in tooltip_data(sample_jobs)
        ],
        tooltip_duration=None,
    ),

    html.Div(children=[
        html.Button(
            id='submit-button', 
            n_clicks=0, 
            children='Submit',
            style={
                'backgroundColor': '#FFFFFF',
                'font-family': 'Open Sans',
            }
        )],
        style={
            'padding': '22px',
        }
    ),
    

    html.Div(
        id = 'output-div',
        children=[
            html.Div(
                style={
                    'textAlign': 'left',
                    'color': '#FFFFFF',
                    'fontSize': 40,
                    'margin': '50px 0px 0px 25px'
                }, 
                children='''Your Results:'''
            ),

            html.Div(
                style={
                    'textAlign': 'left',
                    'color': '#FFFFFF',
                    'margin': '50px 22px 0px 25px',
                    'fontSize': 16,
                }, 
                children=['Under each of the four option tabs you can see a different list of 10 job recommendations based on your selections. The jobs are ordered with most relevant first.',
                            html.Br(), html.Br(), 'Three different algorithms, A, B and C, are being used to recommend you jobs based on your selections.',
                            html.Ul(children=[
                                    html.Li('Algorithm A is used by Option 1 and Option 2'),
                                    html.Li('Algorithm B is used by Option 3'),
                                    html.Li('Algorithm C is used by Option 4'),
                            ]),
                            html.Br(), 'The Compare Algorithms tab is to allow you to easily compare the recommendations produced by these three algorithms.',
                            html.Br(), html.Br(), 'Each Option has a different associated explanation:',
                            html.Br(),
                            html.Ul(children=[
                                    html.Li('Option 1: For each job, it explains which of the two jobs from the ones you selected were most influencial in Algorithm A recommending that job.'),
                                    html.Li('Option 2: For each job, it provides you with an explanation of the average qualifications of users who have previously applied to that job.'),
                                    html.Li('Option 3: It provides you with an explanation of how Algorithm B produces its results, which can be found in the box above the recommendation list.'),
                                    html.Li('Option 4: It provides you with an explanation of how Algorithm C produces its results, which can again be found in the box above the recommendation list. You will then find tailored explanations as to why you are being recommended each job.')
                            ]),
                        ]
            ),
             html.Div(
                style={
                    'margin': '25px 22px 0px 25px',
                    'color': '#FFFFFF',
                    'fontSize': 15,
                }, 
                children=['Hover your mouse over the text in the Description and Requirements columns to view more information about each job.']
            ),

            html.Div(children=[
                dcc.Dropdown(
                    id = 'tables-options',
                    options=[
                        {'label': 'Option 1', 'value': 'mf-job-explanation'},
                        {'label': 'Option 2', 'value': 'mf-db-explanation'},
                        {'label': 'Option 3', 'value': 'ucf'},
                        {'label': 'Option 4', 'value': 'icf'},
                        {'label': 'Compare Algorithms', 'value': 'compare'}
                    ],
                    value='mf-job-explanation',
                )],
                style={
                    'padding': '22px',
                    'width': '30%'
                }
            ),
            dcc.Loading(
                id='loading-state',
                type='default',
                children=html.Div(id='output-state')
            ),
        ], 
        style={
            'display': 'none',
        }
    ),
])


@app.callback(
    Output('sample-jobs', 'style_data_conditional'),
    [Input('sample-jobs', 'selected_rows')]
)
def update_styles(selected_rows):
    return [{
        'if': { 'row_index': i },
        'background_color': '#D2F3FF'
    } for i in selected_rows]


@app.callback(
    Output('output-div', 'style'),
    [Input('submit-button', 'n_clicks')]
)
def show_checklist(n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-button' in changed_id:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('output-state', 'children'),
    [Input('tables-options', 'value')],
    [Input('submit-button', 'n_clicks')],
    [State('sample-jobs', 'selected_rows')], 
)
def display_tables(value, n_clicks, selected_rows):
    if (selected_rows == []):
        return None
    else: 
        selected_jobs = sample_jobs.iloc[selected_rows]
        mf_recommendations, explanations = mf_recommend_jobs(selected_jobs, job_hashmap)
        if (value == 'mf-job-explanation'):
            return(generate_table(mf_map_jobs(mf_recommendations, explanations)))
        elif (value == 'mf-db-explanation'):
            return(generate_table(db_explanation_map_jobs(mf_recommendations)))
        elif (value == 'icf'):
            job_cf_recommendations, explanation = job_cf_recommend_jobs(job_hashmap, selected_jobs)
            return( html.Div(
                        id='icf_intro_description', 
                        style={
                            'margin': '22px 30px 0px 30px',
                            'color': '#ffffff',
                            'font-family': 'Open Sans',
                            'fontSize': 16,
                            'border': '2px white solid',
                            'padding': '5px 0px 5px 10px',
                        },
                        children=['Two jobs are considered similar if many users who applied to one also applied to the other. The system recommends you similar jobs to those you selected.']
                    ),
                    generate_table(item_cf_map_jobs(job_cf_recommendations, explanation)) )
        elif (value == 'ucf'):
            user_cf_recommendations, nearest_neighbours = user_cf_recommend_jobs(user_hashmap, job_hashmap, selected_jobs)
            intro_description = user_information(nearest_neighbours)
            return( html.Div(
                        id='ucf_intro_description', 
                        style={
                            'margin': '22px 30px 0px 30px',
                            'color': '#ffffff',
                            'font-family': 'Open Sans',
                            'fontSize': 16,
                            'border': '2px white solid',
                            'padding': '5px 0px 5px 10px',
                        },
                        children=intro_description
                    ),   
                    generate_table_without_explanations(user_cf_map_jobs(user_cf_recommendations)) )
        else:
            algorithm_a_recs = mf_map_jobs(mf_recommendations, explanations)['Title']

            user_cf_recommendations, _ = user_cf_recommend_jobs(user_hashmap, job_hashmap, selected_jobs)
            algorithm_b_recs = user_cf_map_jobs(user_cf_recommendations)['Title']

            job_cf_recommendations, explanation = job_cf_recommend_jobs(job_hashmap, selected_jobs)
            algorithm_c_recs = item_cf_map_jobs(job_cf_recommendations, explanation)['Title']

            return( generate_comparison_table(algorithm_a_recs, algorithm_b_recs, algorithm_c_recs) )


if __name__ == '__main__':
    app.run_server(debug=True)