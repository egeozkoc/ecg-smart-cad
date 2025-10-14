import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
from plotly.subplots import make_subplots
import scipy.signal as signal
# import dash_auth
import glob
import xml.etree.ElementTree as et


# Database
# 1) usernames/passwords
# 2) 10 sec data
# 3) Philips median beat data
# 4) Philips fiducial markers


# Keep this out of source code repository - save in a file or a database
# VALID_USERNAME_PASSWORD_PAIRS = {'hello': 'world', '':''}

app = dash.Dash(__name__)
# server = app.server

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

# Dataset ########################################
names = glob.glob('/data/*.npy')
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'rms']
dict1 = {}
for idx, name in zip(np.arange(len(names)), names):
    name = name[5:-4]
    name = name.lower()
    dict1[str(idx)] = name
##################################################

# Layout #########################################
app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'ECG SMART', style = {'textAlign':'center', 'marginTop':40,'marginBottom':40}),
    dcc.Dropdown(id = 'dropdown1', options = dict1, value = '0'),
    dcc.RadioItems(id = 'dropdown2', options = {'0':'10 Seconds', '1':'Median Beat'}, value='0', inline=True),
    dcc.RadioItems(id = 'radio1', options = ['Local', 'Global'], value = 'Local', inline=True),
    dcc.Checklist(id = 'list1', options = {'0':'Fiducials'}),
    dcc.Checklist(id = 'list2', options = {'0':'P', '1': 'QRS', '2': 'T'}, value=['0','1','2'], inline=True),
    dcc.Checklist(id = 'list1b', options = {'0': 'Beats'}),
    dcc.RadioItems(id = 'list3', options = {'0':'Raw', '1':'Filtered'}, value='0', inline=True),
    dcc.Checklist(id='list4', options = {'0':'R Peaks'}),
    dcc.Graph(id = 'line_plot1'),
    dcc.Store(id='median-beat'),
    dcc.Store(id='raw-data'),
    dcc.Store(id='filtered-data'),
    dcc.Store(id='beats'),
    dcc.Store(id='fiducials'),
    dcc.Store(id='rpeaks')])
##################################################

def load_file(filename):
    data = np.load(filename, allow_pickle=True).item()
    return data

# Callbacks ######################################
@app.callback(Output(component_id='list1', component_property= 'style'),
              Output(component_id='list1b', component_property= 'style'),
              Output(component_id='list3', component_property= 'style'),
              Output(component_id='list4', component_property= 'style'),
              [Input(component_id='dropdown2', component_property= 'value')],
              [Input(component_id='radio1', component_property= 'value')])
def show_plot(dropdown2_value, radio1_value):
    if dropdown2_value == '0': # 10s
        disp_list1 = {'display': 'none'}
        disp_list1b = {'display': 'none'}
        disp_list3 = {'display': 'block'}
        disp_list4 = {'display': 'block'}
    elif dropdown2_value == '1': # median beat
        disp_list1 = {'display': 'block'}
        disp_list3 = {'display': 'none'}
        disp_list4 = {'display': 'none'}
        if radio1_value == 'Local':
            disp_list1b = {'display': 'block'}
        elif radio1_value == 'Global':
            disp_list1b = {'display': 'none'}
    return disp_list1, disp_list1b, disp_list3, disp_list4

@app.callback(Output(component_id='median-beat', component_property='data'),
              Output(component_id='raw-data', component_property='data'),
              Output(component_id='filtered-data', component_property='data'),
              Output(component_id='fiducials', component_property='data'),
              Output(component_id='beats', component_property='data'),
              Output(component_id='rpeaks', component_property='data'),
              [Input(component_id='dropdown1', component_property='value')])
def calculate_features(dropdown_value1):
    dataset = load_file(names[int(dropdown_value1)])

    data = dataset['ecg_10sec']
    data_filtered = dataset['ecg_10sec_filtered']
    median_beat = dataset['ecg_median']
    beats = dataset['beats']
    rpeaks = dataset['rpeaks']
    fiducials = dataset['fiducials']

    return median_beat/1000, data/1000, data_filtered/1000, fiducials, beats/1000, rpeaks

@app.callback(Output(component_id='list2', component_property='style'),
              [Input(component_id='list1', component_property='value')],
              [Input(component_id='list1', component_property='style')])
def show_list(list_value,list_style):
    if list_value != None and ('0' in list_value) and list_style == {'display': 'block'}:
        disp_list = {'display': 'block'}
    else:
        disp_list = {'display': 'none'}
    return disp_list

@app.callback(Output(component_id='line_plot1', component_property= 'figure'),
              [Input(component_id='dropdown2', component_property= 'value')],
              [Input(component_id='list1', component_property= 'value')],
              [Input(component_id='list1b', component_property= 'value')],
              [Input(component_id='list2', component_property= 'value')],
              [Input(component_id='list3', component_property= 'value')],
              [Input(component_id='list4', component_property= 'value')],
              [Input(component_id='radio1', component_property= 'value')],
              [Input(component_id='median-beat', component_property='data')],
              [Input(component_id='raw-data', component_property='data')],
              [Input(component_id='filtered-data', component_property='data')],
              [Input(component_id='beats', component_property='data')],
              [Input(component_id='fiducials', component_property='data')],
              [Input(component_id='rpeaks', component_property='data')])
def plot_ecg(dropdown_value2, list_value, list_value1b, list_value2, list_value3, list_value4, radio_value1, median_beat, data_raw, data_filtered, beats, fiducials, rpeaks):
    data_raw = np.asarray(data_raw)
    data_filtered = np.asarray(data_filtered)
    beats = np.asarray(beats)
    fs = 500

    if dropdown_value2 == '0' and radio_value1 == 'Local':
        fig = go.Figure()
        offset = 0
        for lead in range(12):
            if list_value3 == '0':
                fig.add_trace(go.Scatter(x=np.arange(5000), y=data_raw[lead,:] - offset, line = dict(color='rgba(0, 0, 0, 1)')))
            elif list_value3 == '1':
                fig.add_trace(go.Scatter(x=np.arange(5000), y=data_filtered[lead,:] - offset, line = dict(color='rgba(0, 0, 0, 1)')))
            fig.add_annotation(dict(font=dict(color='black',size=30),x=10,y=.5 - offset,showarrow=False,text=leads[lead], xanchor='left'))
            offset += 3
        if list_value4 != None and '0' in list_value4:
            for peak in rpeaks:
                fig.add_shape(type="line", x0=peak, y0=-34.5, x1=peak, y1=1.5, line=dict(color="rgba(0,0,200,1)",width=2))

        width = 1500
        height = int(width/50*72)
        fig.update_layout(plot_bgcolor='white', title='10s ECG', width = width, height = height, showlegend=False)
        fig.update_xaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.2*fs,showgrid=True, gridwidth=2, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', minor={'dtick':0.04*fs, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[0,10*fs])
        fig.update_yaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.5,showgrid=True, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', gridwidth=2, minor={'dtick':0.1, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[-34.5,1.5])
        print('plot_10')
        return fig
    
    elif dropdown_value2 == '0' and radio_value1 == 'Global':
        fig = go.Figure()
        for lead in range(12):
            if list_value3 =='0':
                fig.add_trace(go.Scatter(x=np.arange(5000), y=data_raw[lead,:], line = dict(color='rgba(0, 0, 0, 1)')))
            elif list_value3 == '1':
                fig.add_trace(go.Scatter(x=np.arange(5000), y=data_filtered[lead,:], line = dict(color='rgba(0, 0, 0, 1)')))

        if list_value4 != None and '0' in list_value4:
            for peak in rpeaks:
                fig.add_shape(type="line", x0=peak, y0=-2, x1=peak, y1=2, line=dict(color="rgba(0,0,200,1)",width=2))

        width = 1500
        height = int(width/50*13)
        fig.update_layout(plot_bgcolor='white', title='10s ECG', width = width, height = height, showlegend=False)
        fig.update_xaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.2*fs,showgrid=True, gridwidth=2, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', minor={'dtick':0.04*fs, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[0,10*fs])
        fig.update_yaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.5,showgrid=True, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', gridwidth=2, minor={'dtick':0.1, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[-2,2])
        print('plot_10')
        return fig
    


    elif dropdown_value2 == '1' and radio_value1 == 'Local':
        median_beat = np.asarray(median_beat)
        qrs_onsets_local = np.asarray(fiducials['qrs_onsets_local'])
        qrs_offsets_local = np.asarray(fiducials['qrs_offsets_local'])
        p_onset = np.asarray(fiducials['p_onset'])
        p_offset = np.asarray(fiducials['p_offset'])
        t_onset = np.asarray(fiducials['t_onset'])
        t_offset = np.asarray(fiducials['t_offset'])

        fig = go.Figure()
        offset = 0
        for lead in range(0,12,3):
            if list_value1b != None and '0' in list_value1b:
                for beat in range(len(beats)):
                    fig.add_trace(go.Scatter(x = np.arange(0,0.8*fs), y=beats[beat,lead,:] - offset, line = dict(color='rgba(150, 150, 150, 0.5)')))
                    fig.add_trace(go.Scatter(x = np.arange(0.8*fs,0.8*fs*2), y=beats[beat,lead+1,:] - offset, line = dict(color='rgba(150, 150, 150, 0.5)')))
                    fig.add_trace(go.Scatter(x = np.arange(0.8*fs*2,0.8*fs*3), y=beats[beat,lead+2,:] - offset, line = dict(color='rgba(150, 150, 150, 0.5)')))


            fig.add_trace(go.Scatter(x=np.arange(0.8*fs), y=median_beat[lead,:] - offset, line = dict(color='rgba(0, 0, 0, 1)')))
            fig.add_annotation(dict(font=dict(color='black',size=30),x=10,y=.5 - offset,showarrow=False,text=leads[lead], xanchor='left'))
            fig.add_trace(go.Scatter(x=np.arange(0.8*fs,0.8*fs*2), y=median_beat[lead+1] - offset, line = dict(color='rgba(0, 0, 0, 1)')))
            fig.add_annotation(dict(font=dict(color='black',size=30),x=0.8*fs+10,y=.5 - offset,showarrow=False,text=leads[lead+1], xanchor='left'))
            fig.add_trace(go.Scatter(x=np.arange(0.8*fs*2,0.8*fs*3), y=median_beat[lead+2] - offset, line = dict(color='rgba(0, 0, 0, 1)')))
            fig.add_annotation(dict(font=dict(color='black',size=30),x=0.8*fs*2+10,y=.5 - offset,showarrow=False,text=leads[lead+2], xanchor='left'))
            
            if list_value != None and '0' in list_value:
                if list_value2 !=None and '0' in list_value2:
                    fig.add_shape(type='line', x0=p_onset+0, y0=-1 - offset, x1=p_onset+0, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=p_offset+0, y0=-1 - offset, x1=p_offset+0, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=p_onset+0.8*fs, y0=-1 - offset, x1=p_onset+0.8*fs, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=p_offset+0.8*fs, y0=-1 - offset, x1=p_offset+0.8*fs, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=p_onset+0.8*fs*2, y0=-1 - offset, x1=p_onset+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=p_offset+0.8*fs*2, y0=-1 - offset, x1=p_offset+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(200,0,200,1)',width=2))
                if list_value2 !=None and '1' in list_value2:
                    fig.add_shape(type='line', x0=qrs_onsets_local[lead]+0, y0=-1 - offset, x1=qrs_onsets_local[lead]+0, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=qrs_offsets_local[lead]+0, y0=-1 - offset, x1=qrs_offsets_local[lead]+0, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=qrs_onsets_local[lead+1]+0.8*fs, y0=-1 - offset, x1=qrs_onsets_local[lead+1]+0.8*fs, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=qrs_offsets_local[lead+1]+0.8*fs, y0=-1 - offset, x1=qrs_offsets_local[lead+1]+0.8*fs, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=qrs_onsets_local[lead+2]+0.8*fs*2, y0=-1 - offset, x1=qrs_onsets_local[lead+2]+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                    fig.add_shape(type='line', x0=qrs_offsets_local[lead+2]+0.8*fs*2, y0=-1 - offset, x1=qrs_offsets_local[lead+2]+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(0,0,200,1)',width=2))
                if list_value2 !=None and '2' in list_value2:
                    fig.add_shape(type='line', x0=t_onset+0, y0=-1 - offset, x1=t_onset+0, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))
                    fig.add_shape(type='line', x0=t_offset+0, y0=-1 - offset, x1=t_offset+0, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))
                    fig.add_shape(type='line', x0=t_onset+0.8*fs, y0=-1 - offset, x1=t_onset+0.8*fs, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))
                    fig.add_shape(type='line', x0=t_offset+0.8*fs, y0=-1 - offset, x1=t_offset+0.8*fs, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))
                    fig.add_shape(type='line', x0=t_onset+0.8*fs*2, y0=-1 - offset, x1=t_onset+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))
                    fig.add_shape(type='line', x0=t_offset+0.8*fs*2, y0=-1 - offset, x1=t_offset+0.8*fs*2, y1=1 - offset, line=dict(color='rgba(0,150,0,1)',width=2))

            offset += 3

        fig.add_shape(type="line", x0=0.8*fs, y0=-10.5, x1=0.8*fs, y1=1.5, line=dict(color="black",width=2))
        fig.add_shape(type="line", x0=0.8*fs*2, y0=-10.5, x1=0.8*fs*2, y1=1.5, line=dict(color="black",width=2))

        width = 750
        height = width*2

        fig.update_layout(plot_bgcolor='white', title='Median Beat', width = width, height = height, showlegend=False)
        fig.update_xaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.2*fs,showgrid=True, gridwidth=2, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', minor={'dtick':0.04*fs, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[0,0.8*fs*3])
        fig.update_yaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.5,showgrid=True, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', gridwidth=2, minor={'dtick':0.1, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[-10.5,1.5])
        return fig
    
    elif dropdown_value2 == '1' and radio_value1 == 'Global':
        median_beat = np.asarray(median_beat)
        p_onset = np.asarray(fiducials['p_onset'])
        p_offset = np.asarray(fiducials['p_offset'])
        qrs_onset = np.asarray(fiducials['qrs_onset'])
        qrs_offset = np.asarray(fiducials['qrs_offset'])
        t_onset = np.asarray(fiducials['t_onset'])
        t_offset = np.asarray(fiducials['t_offset'])

        fig = go.Figure()
        offset = 0
        for lead in range(12):
            fig.add_trace(go.Scatter(x=np.arange(0.8*fs), y=median_beat[lead,:] - offset, mode='lines'))
        if list_value != None and '0' in list_value:
            if list_value2 !=None and '0' in list_value2:
                fig.add_shape(type='line', x0=p_onset+0, y0=-1, x1=p_onset+0, y1=1, line=dict(color='rgba(200,0,200,1)',width=2))
                fig.add_shape(type='line', x0=p_offset+0, y0=-1, x1=p_offset+0, y1=1, line=dict(color='rgba(200,0,200,1)',width=2))
            if list_value2 !=None and '1' in list_value2:
                fig.add_shape(type='line', x0=qrs_onset+0, y0=-1, x1=qrs_onset+0, y1=1, line=dict(color='rgba(0,0,200,1)',width=2))
                fig.add_shape(type='line', x0=qrs_offset+0, y0=-1, x1=qrs_offset+0, y1=1, line=dict(color='rgba(0,0,200,1)',width=2))
            if list_value2 !=None and '2' in list_value2:
                fig.add_shape(type='line', x0=t_onset+0, y0=-1, x1=t_onset+0, y1=1, line=dict(color='rgba(0,150,0,1)',width=2))
                fig.add_shape(type='line', x0=t_offset+0, y0=-1, x1=t_offset+0, y1=1, line=dict(color='rgba(0,150,0,1)',width=2))

        width = 500
        height = width*2

        fig.update_layout(plot_bgcolor='white', title='Median Beat', width = width, height = height, showlegend=False)
        fig.update_traces(line_color='black')
        fig.update_xaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.2*fs,showgrid=True, gridwidth=2, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', minor={'dtick':0.04*fs, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[0,0.8*fs])
        fig.update_yaxes(showticklabels = False, gridcolor='rgba(255,0,0,0.5)', dtick=0.5,showgrid=True, zeroline=True, zerolinecolor='rgba(255,0,0,0.5)', gridwidth=2, minor={'dtick':0.1, 'gridwidth': 0.05, 'gridcolor':'rgba(255,0,0,0.2)', 'showgrid':True}, range=[-2,2])
        return fig


##################################################

if __name__ == '__main__':
    app.run_server()
