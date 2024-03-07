import dash 
import os
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from dotenv import load_dotenv

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sqlalchemy import create_engine

load_dotenv()

# Form a connection with the server
engine = create_engine(os.getenv('CONNECTOR'))

#WER of the jasmin dataset
WER_stats = pd.read_sql_query(
        'SELECT * FROM WER_CER',
        engine
    )
WER_stats['Age'] = WER_stats['Age'].astype(float)

#List of the files
u_list = WER_stats['Root']
u_list = u_list.to_list()
options = [html.Option(value=x) for x in u_list if x in ['fn100230','fn100067','fn100068']]

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '1%',
    'margin-right': '1%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'margin-top': '1%',
    'margin-bottom': '2%',
}


# Content first row
content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        # html.H4("Inputs", style={'text-align':'center'}),
                        html.P("Gender"),
                        html.P(dcc.RadioItems(
        ['All', 'Male' , 'Female'],
        'All',
        id='Geslacht',
        inline=True,
        inputStyle={'margin-left': '10px'}
        ),
        ),html.P("Region"),
                        html.P(dcc.RadioItems(
        ['All', 'Flemish' , 'Dutch'],
        'All',
        id='Source',
        inline=True,
        inputStyle={'margin-left': '10px'}
        ),
        ),html.P("Age range"),
                        html.P(dcc.RangeSlider(4, 80, marks=None, value=[4, 80],step= 1,tooltip={"placement": "bottom", "always_visible": True}, id= 'Leeftijd')),
                        html.P("Error Rate Type"),
                        html.P(dcc.RadioItems(
        ['WER' , 'CER'],
        'WER',
        id='Error',
        inline=True,
        inputStyle={'margin-left': '10px'}
        ),
        ),
                        html.P("Language Proficiency"),
                        html.P(dcc.RadioItems(
        ['All','Native' , 'Non-Native'],
        'All',
        id='native',
        inline=True,
        inputStyle={'margin-left': '10px'}
        ),
        ),
                        html.P("Type of session"),
                        html.P(dcc.RadioItems(
        ['All','HMI' , 'Reading'],
        'All',
        id='data',
        inline=True,
        inputStyle={'margin-left': '10px'}
        ),
        )
        
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dcc.Graph(id='G1'), md=6
        ),
    dbc.Col(
        dcc.Graph(id='ER_info'), md=3
        )
])

# Content row filler
filler = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H3("Information per session "),

                    ]
                )
            ]
        ),
        md=12,
        style= CARD_TEXT_STYLE
    )
])
# Content row 2
content_second_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4("LM scores"),
                        html.P("You can select the session you want to explore in the text field below. The graph to the right depicts the LM scores of the selected file. It also updates the chart below with the WER and CER per sentence."),
                        html.Datalist(id="files_opt", children=options),
                        html.P(dcc.Input(id="input2", type="text", placeholder="Select your file", list= 'files_opt' ,debounce=True)),
                    ]
                )
            ]
        ),
        md=4
    ),
    dbc.Col(
         dcc.Graph(id='G2'), md=8
        ),
])
# Content row 3
content_third_row = dbc.Row([
    dbc.Col(
         dcc.Graph(id='G3'), md=10
        ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4("Sentence analysis"),
                        html.P("The graph to the left depicts the WER and the CER per sentence. You can hover your mouse to discover more information about the evaluated sentence." )
                    ]
                )
            ]
        ),
        md=2
    ),
])

# True vs predicted
sentence = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H3("Ground truth"),
                        html.P(id='truth')

                    ]
                )
            ]
        ),
        md=6,
        style= CARD_TEXT_STYLE
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H3("Predicted"),
                        html.P(id='predicted')

                    ]
                )
            ]
        ),
        md=6,
        style= CARD_TEXT_STYLE
    )
])

# Model info
M_info = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H3("Model information"),
                        html.P("RTL wants to use AI to automatically generate subtitles. To improve this process, we are investigating whether the model recognizes specific speakers less well than others. If so, the model should be modified to better recognize these speakers. The model in question was shown above, it is an open-source model based on Facebook's W2V model. This model uses a wave 2 vector (W2V) encoder to convert audio to text and is trained on the Corpus Gesproken Nederlands (CGN). The CGN contains more than 900 hours of Dutch spoken data! The testing was done on the Jasmin dataset, which is an extension of the CGN of roughly 100 hours. What makes this dataset special is the variation it contains so it can be tested to see if there is a bias in the model.")

                    ]
                )
            ]
        ),
        md=12,
        style= CARD_TEXT_STYLE
    )
])

# Content samenvatten
content = html.Div(
    [
        html.H3('Interactive Model Card', style=TEXT_STYLE),
        html.Hr(),
        content_first_row,
        filler,
        content_second_row,
        content_third_row,
        sentence,
        M_info
    ],
    style=CONTENT_STYLE
)


app = dash.Dash(external_stylesheets=[dbc.themes.PULSE])
app.layout = html.Div([content])

# Create a dataframe from a SQL query and create a graph
@app.callback(
    Output('G3', 'figure'),
    Input('input2', 'value')
    )
def sql(value):
    dff = pd.read_sql_query(
        'SELECT Root, WER, CER, Truth, Predicted FROM WER_Sentence WHERE Root = "{}"'.format(value),
        engine
    )
    dff[["WER", "CER"]] = dff[["WER", "CER"]].apply(pd.to_numeric)
    fig  = px.line(dff, y = ['WER', 'CER'], hover_data=['Truth', 'Predicted'], markers=True, title='WER & CER per sentence')

    fig.update_xaxes(title_text='Words')
    fig.update_yaxes(title_text='Value',)
    fig.update_layout(
    font_color="black",
    title_font_color="black",
)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    
    return fig

# Create a dataframe from a SQL query and create a boxplot
@app.callback(
    Output('G2', 'figure'),
    Input('input2', 'value')
    )
def sql2(value):
    df = pd.read_sql_query(
        'SELECT * FROM lm_scores WHERE Root = "{}"'.format(value),
        engine
    )
    df[["lm_score"]] = df[["lm_score"]].apply(pd.to_numeric)

    fig= px.box(df, y =['lm_score'] ,title='LM scores' ,hover_data=['word'], points='all')

    fig.update_xaxes(title='y', visible=False, showticklabels=False)
    fig.update_yaxes(title_text='LM Score',)
    fig.update_layout(
    font_color="black",
    title_font_color="black",
)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig

#Callback voor de WER graph
@app.callback(
    Output(component_id='G1', component_property='figure'),
    Input(component_id='Source', component_property='value'),
    Input(component_id='Geslacht', component_property='value'),
    Input(component_id='Leeftijd', component_property='value'),
    Input(component_id='native', component_property='value'),
    Input(component_id='data', component_property='value'),
    Input(component_id='Error', component_property= 'value')
)
def update_wer_graph(Source,Geslacht,Leeftijd,Native,data,error):

    if Geslacht == 'Male':
        WER_stats_new = WER_stats.loc[WER_stats['Gender'] == 'M']
    elif Geslacht == 'Female':
        WER_stats_new = WER_stats.loc[WER_stats['Gender'] == 'F']
    elif Geslacht == 'All':
        WER_stats_new = WER_stats

    if Source == 'Flemish':
        WER_stats_new = WER_stats_new[WER_stats_new['Root'].str.contains("fv")]
    elif Source =='Dutch':
        WER_stats_new = WER_stats_new[WER_stats_new['Root'].str.contains("fn")]
    elif Source =='All':
        WER_stats_new = WER_stats_new

    WER_stats_new = WER_stats_new.loc[WER_stats_new['Age'].between(
                Leeftijd[0],Leeftijd[1]
    )]
    
    if Native == 'All':
        WER_stats_new=WER_stats_new
    elif Native == 'Native':
        WER_stats_new=WER_stats_new.loc[(WER_stats_new['BirthPlace'] == 'NL') | (WER_stats_new['BirthPlace'] == 'BEL' )]
    elif Native == 'Non-Native':
        WER_stats_new = WER_stats_new.drop(WER_stats_new[(WER_stats_new['BirthPlace'] == 'NL') | (WER_stats_new['BirthPlace'] == 'BEL')].index)

    if data == 'Pre-Written':
        WER_stats_new = WER_stats_new[WER_stats_new['Component']=='comp-q']
    elif data =='HMI':
        WER_stats_new = WER_stats_new[WER_stats_new['Component']== 'comp-p']
    elif Source =='All':
        WER_stats_new = WER_stats_new

    WER_stats_new = WER_stats_new.sort_values(by='WER', ascending=False, ignore_index= True)

    if error == 'WER':
            WER_Graph = px.bar(WER_stats_new , y = 'WER',title= 'WER graph', hover_data=['Gender', 'SpeakerID', 'Age' , 'Root', 'CER'], color= 'BirthPlace')
            WER_stats_new['avg'] = WER_stats_new['WER'].mean()
            l = px.line(
            WER_stats_new, y='avg',
            ).update_traces(line_color="red")
            WER_Graph.add_traces(l.data)
    elif error == 'CER':
            WER_Graph = px.bar(WER_stats_new , y = 'CER',title= 'CER graph', hover_data=['Gender', 'SpeakerID', 'Age' , 'Root', 'WER'], color= 'BirthPlace')
            WER_stats_new['avg'] = WER_stats_new['CER'].mean()
            l = px.line(
            WER_stats_new, y='avg',
            ).update_traces(line_color="red")
            WER_Graph.add_traces(l.data)

    figure = WER_Graph

    figure.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    figure.update_layout(
    font_color="black",
    title_font_color="black",
)

    return figure


#Callback voor de Error rate average
@app.callback(
    Output('ER_info', 'figure'),
    Input(component_id='Source', component_property='value'),
    Input(component_id='Geslacht', component_property='value'),
    Input(component_id='Leeftijd', component_property='value'),
    Input(component_id='native', component_property='value'),
    Input(component_id='data', component_property='value'),
    Input(component_id='Error', component_property= 'value')
)
def update_wer_graph(Source,Geslacht,Leeftijd,Native,data ,error):

    if Geslacht == 'Male':
        WER_stats_new = WER_stats.loc[WER_stats['Gender'] == 'M']
    elif Geslacht == 'Female':
        WER_stats_new = WER_stats.loc[WER_stats['Gender'] == 'F']
    elif Geslacht == 'All':
        WER_stats_new = WER_stats

    if Source == 'Flemish':
        WER_stats_new = WER_stats_new[WER_stats_new['Root'].str.contains("fv")]
    elif Source =='Dutch':
        WER_stats_new = WER_stats_new[WER_stats_new['Root'].str.contains("fn")]
    elif Source =='All':
        WER_stats_new = WER_stats_new

    WER_stats_new = WER_stats_new.loc[WER_stats_new['Age'].between(
                Leeftijd[0],Leeftijd[1]
    )]

    if Native == 'All':
        WER_stats_new=WER_stats_new
    elif Native == 'Native':
        WER_stats_new=WER_stats_new.loc[(WER_stats_new['BirthPlace'] == 'NL') | (WER_stats_new['BirthPlace'] == 'BEL' )]
    elif Native == 'Non-Native':
        WER_stats_new = WER_stats_new.drop(WER_stats_new[(WER_stats_new['BirthPlace'] == 'NL') | (WER_stats_new['BirthPlace'] == 'BEL')].index)

    if data == 'Pre-Wrtiten':
        WER_stats_new = WER_stats_new[WER_stats_new['Component']=='comp-q']
    elif data =='HMI':
        WER_stats_new = WER_stats_new[WER_stats_new['Component']== 'comp-p']
    elif Source =='All':
        WER_stats_new = WER_stats_new

    WER_stats_new = WER_stats_new.sort_values(by='WER', ascending=False, ignore_index= True)

    if error == 'WER':
            figure = px.box(WER_stats_new, y ='WER' ,title='WER boxplot')
    elif error == 'CER':
            figure = px.box(WER_stats_new, y ='CER' ,title='CER boxplot')


    y_lower_limit = 0  
    y_upper_limit = 1  
    figure.update_yaxes(range=[y_lower_limit, y_upper_limit])    

            
    figure.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    
    figure.update_layout(
    font_color="black",
    title_font_color="black",
)
    return figure

# truth
@app.callback(
    Output('truth', 'children'),
    Input('input2', 'value')
    )
def sql3(value):
    df = pd.read_sql_query(
        'SELECT Truth FROM WER_Sentence WHERE Root = "{}"'.format(value),
        engine
    )
    one_string = ' '.join(df['Truth'].tolist())

    return one_string

# predicted
@app.callback(
    Output('predicted', 'children'),
    Input('input2', 'value')
    )
def sql4(value):
    df = pd.read_sql_query(
        'SELECT Predicted FROM WER_Sentence WHERE Root = "{}"'.format(value),
        engine
    )
    one_string = ' '.join(df['Predicted'].tolist())

    return one_string

# Run the application
if __name__ == '__main__':
    app.run_server(host= '0.0.0.0',port=os.environ['PORT'] , debug= False)
    
