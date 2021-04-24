import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

def tooltip_data(sample_jobs):
    '''
    Add functionality to hover over the Requirements and Descriptions column to see the full text.
    '''
    dictionary = []
    for row in sample_jobs.to_dict('rows'):
        new_row = {column: value for column, value in row.items() if (column == 'Description' or column == 'Requirements')}
        dictionary = dictionary + [new_row]
    return dictionary


def generate_table_without_explanations(df):
    '''
    Generate table without the explanations for the User-Global model.
    '''
    recommendations = df

    return( dash_table.DataTable(
                id='recommendations',
                columns=[
                    {"name": 'Title', "id": 'Title'}, 
                    {"name": 'Description', "id": 'Description'}, 
                    {"name": 'Requirements', "id": 'Requirements'}, 
                    {"name": 'City', "id": 'City'}, 
                    {"name": 'State', "id": 'State'}, 
                    {"name": 'Country', "id": 'Country'}
                ], 
                data=recommendations.to_dict('records'),
                style_data={
                    'whiteSpace': 'nowrap',
                },
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px 5px 5px 20px', 
                    'textOverflow': 'clip',
                    'overflow': 'hidden',
                    'maxWidth': '70px', 
                    'minWidth': '70px',
                    'width': '70px',
                    'minHeight': '45px', 
                    'height': '45px', 
                },
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': 'Title',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                            'fontWeight': 'bold'
                    },
                    {
                        'if': {
                            'column_id': 'Requirements',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                    },
                    {
                        'if': {
                            'column_id': 'Description',
                        },
                            'minWidth': '250px',
                            'maxWidth': '250px',
                            'width': '250px',
                    },
                    {
                        'if': {
                            'column_id': 'City',
                        },
                            'minWidth': '90px',
                            'maxWidth': '90px',
                            'width': '90px',
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
                    } for row in tooltip_data(recommendations)
                ],
                tooltip_duration=None,
            ))


def generate_table(df):
    '''
    Generate table with explanations column for Item-Only, User-Only and Item-Global models.
    '''
    recommendations = df

    return( dash_table.DataTable(
                id='recommendations',
                columns=[
                    {"name": 'Title', "id": 'Title'}, 
                    {"name": 'Explanations', "id": 'Explanations'},
                    {"name": 'Description', "id": 'Description'}, 
                    {"name": 'Requirements', "id": 'Requirements'}, 
                    {"name": 'City', "id": 'City'}, 
                    {"name": 'State', "id": 'State'}, 
                    {"name": 'Country', "id": 'Country'},
                ], 
                data=recommendations.to_dict('records'),
                style_cell_conditional=[
                    {
                        'if': {
                            'column_id': 'Explanations',
                        },
                        'minWidth': '235px',
                        'maxWidth': '235px',
                        'width': '235px',
                        'whiteSpace': 'normal',
                        'backgroundColor': '#c9eaf7 ',
                    },
                    {
                        'if': {
                            'column_id': 'Title',
                        },
                        'minWidth': '200px',
                        'maxWidth': '200px',
                        'width': '200px',
                        'fontWeight': 'bold',
                    },
                    {
                        'if': {
                            'column_id': 'Requirements',
                        },
                            'minWidth': '200px',
                            'maxWidth': '200px',
                            'width': '200px',
                    },
                    {
                        'if': {
                            'column_id': 'Description',
                        },
                            'minWidth': '200px',
                            'maxWidth': '200px',
                            'width': '200px',
                    },
                    {
                        'if': {
                            'column_id': 'City',
                        },
                            'minWidth': '90px',
                            'maxWidth': '90px',
                            'width': '90px',
                    },
                ],
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px 5px 5px 20px', 
                    'textOverflow': 'clip',
                    'overflow': 'hidden',
                    'maxWidth': '75px', 
                    'minWidth': '75px',
                    'width': '75px',
                    'minHeight': '45px', 
                    'maxHeight': '45px', 
                    'height': '45px', 
                },
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
                    } for row in tooltip_data(recommendations)
                ],
                tooltip_duration=None,
            ))


def generate_comparison_table(algorithm_a_recs, algorithm_b_recs, algorithm_c_recs):
    '''
    Generate table to compare the job titles recommended from the three algorithms.
    '''
    data = { 'Algorithm-A': algorithm_a_recs.to_numpy(), 'Algorithm-B': algorithm_b_recs.to_numpy(), 'Algorithm-C': algorithm_c_recs.to_numpy() }
    comparions = pd.DataFrame.from_dict(data)

    return( dash_table.DataTable(
                id='comparion',
                columns=[
                    {"name": 'Algorithm-A', "id": 'Algorithm-A'}, 
                    {"name": 'Algorithm-B', "id": 'Algorithm-B'},
                    {"name": 'Algorithm-C', "id": 'Algorithm-C'}, 
                ], 
                data=comparions.to_dict('records'),
                style_cell={
                    'textAlign': 'left',
                    'font-family': 'Open Sans',
                    'fontSize': 13,
                    'backgroundColor': '#d9ecf3',
                    'padding': '5px 5px 5px 35px', 
                    'textOverflow': 'ellipsis',
                    'overflow': 'hidden',
                    'maxWidth': '190px', 
                    'minWidth': '190px',
                    'width': '190px',
                    'minHeight': '45px', 
                    'maxHeight': '45px', 
                    'height': '45px', 
                },
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
            ))