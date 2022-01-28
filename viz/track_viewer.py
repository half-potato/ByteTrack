import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.track_parser import TrackParser

import dash
from dash import Dash, dcc, html, Input, Output, no_update, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


app = Dash(__name__)

data_root = Path("/data/MOT17/train")
seq_names = [seq_path.name for seq_path in data_root.iterdir()]

parsers = {seq_name:
    TrackParser(data_root, seq_name, "mot", max_missing_frames=2).load_annotations()
    for seq_name in seq_names}

g_seq_name = seq_names[0]

def create_seq_dropdown():
    options = [{'label': seq_name, 'value': seq_name} for seq_name in seq_names]
    return [
        dcc.Dropdown(
            id='seq-dropdown',
            options=options,
            value=options[0]['value'],
            style={'width': '100%', 'margin-right': '0px'},
        )
    ]

@app.callback(
    Output('main-view', 'children'),
    Input('seq-dropdown', 'value'),
)
def select_seq(seq_name):
    global g_seq_name
    g_seq_name = seq_name if seq_name is not None else seq_names[0]
    return create_main_view()

@app.callback(
    Output('main-plot-container', 'children'),
    Input('frame-slider', 'value'),
)
def create_plot(frame_id):
    global g_seq_name
    fig = parsers[g_seq_name].plot(frame_id)
    return [dcc.Graph(id="main-plot", figure=fig, clear_on_unhover=True)]

def create_main_view():
    global g_seq_name
    return [html.Div(children=[
        html.Div(children=[
            dcc.Slider(id='frame-slider', min=1, max=len(parsers[g_seq_name].frames), value=1, step=1),
        ], id='control-container', style={'box-sizing': 'border-box', 'width': '100%', 'height': '10%'}),
        html.Div(children=[
            dcc.Graph(id="main-plot", figure=parsers[g_seq_name].plot(1), clear_on_unhover=True)
            ],
            id='main-plot-container',
            style={'box-sizing': 'border-box', 'width': '100%', 'height': '90%'}),
    ])]

app.layout = html.Div([
    html.Div(children=create_seq_dropdown()),
    html.Div(children=create_main_view(), id='main-view'),
], id='main', className="container")

app.run_server(debug=True)