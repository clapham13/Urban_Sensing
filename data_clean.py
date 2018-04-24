# Let's define some useful operations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplleaflet

def concat_acc_gyro(acc_df, gyro_df):
    '''
    Takes in gyro and acceleration data, groups them into the nearest 
    1/10th of a second; returns a concatenation of the dataframes.
    '''
    acc_df['time_tick'] = acc_df.time_tick.round(1)
    gyro_df['time_tick'] = gyro_df.time_tick.round(1)
    acc_df = acc_df.groupby('time_tick').mean()
    gyro_df = gyro_df.groupby('time_tick').mean()
    result = pd.concat([acc_df, gyro_df], axis=1, join_axes=[acc_df.index])
    return result

def concat_gpx_trial_data(trial_data, gpx_df):
    '''
    Performs some transformations on the gpx data and returns a merge with trial data
    '''
    gpx_df.drop(0, inplace=True)
    gpx_df.set_index('Time', inplace=True)
    gpx_df.drop('Speed', axis=1, inplace=True)
    gpx_df = gpx_df.reindex(trial_data.index)
    gpx_df.interpolate(inplace=True)
    result = pd.concat([trial_data, gpx_df], axis=1, join_axes=[trial_data.index])
    return result.dropna()

def merge_data(TRIAL_NUMBER):
    '''
    performs needed operations, and returns a merged dataframe of all of the data
    '''
    # import and examine the gyroscope data
    gyro = pd.read_csv('data/' + str(TRIAL_NUMBER) + '-gyr.csv')

    # import and examine the gyroscope data
    acc = pd.read_csv('data/' + str(TRIAL_NUMBER) + '-acc.csv')

    # concatenate the acc and gyro
    trial_data = concat_acc_gyro(acc, gyro)

    # import gpx data and put into dataframe
    gpx_df = gpx_to_pandas('data/' + str(TRIAL_NUMBER) + '.gpx')
    # convert Time column from timestamp to seconds
    gpx_df['Time'] = [seconds(gpx_df, i) for i in range(len(gpx_df))] 

    # merge it all together
    trial_data = concat_gpx_trial_data(trial_data, gpx_df)
    
    return trial_data

def gpx_to_pandas(filename):
    '''
    Takes in a gpx file (as a string) and returns a pandas dataframe with columns: 
    'Longitude', 'Latitude', 'Altitude', 'Time', 'Speed'
    '''
    import gpxpy
    
    gpx = gpxpy.parse(open(filename))
    
    track = gpx.tracks[0]
    segment = track.segments[0]
    data = []
    
    segment_length = segment.length_3d()
    for point_idx, point in enumerate(segment.points):
        data.append([point.longitude, point.latitude,
                     point.elevation, point.time, segment.get_speed(point_idx)])
        
    columns = ['Longitude', 'Latitude', 'Altitude', 'Time', 'Speed']
    df = pd.DataFrame(data, columns=columns)
    return (df)

def seconds(gpx_df, i):
    diff = round((gpx_df.Time[i] - gpx_df.Time[0]).total_seconds(), 0)
    return diff

def plot_gpx(df):
    '''
    Takes in a gpx dataframe and plots on mplleaflet background
    '''
    fig, ax = plt.subplots(figsize=(5,5))
    df = df.dropna()
    ax.plot(df['Longitude'], df['Latitude'],
            color='darkorange', linewidth=5, alpha=0.5)
    sub = 10
    return mplleaflet.display(fig=fig, tiles='esri_aerial')

def plot_gyro_acc(df):
    '''
    Takes in data from a bike trial run; returns time-series plot of X, Y, & Z axes 
    '''
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)

    x_axis = df.index

    x_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_X_value,
        name = 'X-Axis'
    )
    y_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_Y_value,
        name='Y-Axis'
    )
    z_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_Z_value,
        name='Z-Axis'
    )
    x_acc = go.Scatter(
        x=x_axis,
        y=df.acc_X_value,
        name = 'X-Axis'
    )
    y_acc = go.Scatter(
        x=x_axis,
        y=df.acc_Y_value,
        name='Y-Axis'
    )
    z_acc = go.Scatter(
        x=x_axis,
        y=df.acc_Z_value,
        name='Z-Axis'
    )

    data = [x_gyro, y_gyro, z_gyro, x_acc, y_acc, z_acc]
    return iplot({'data': data, 'layout': {'title': 'Ride Data', 
                                           'font': dict(size=16)}})

def subplots_gyro_acc(df):
    '''
    Takes in data from a bike trial run; returns time-series subplots of X, Y, & Z axes 
    '''
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    from plotly  import  tools
    init_notebook_mode(connected=True)

    x_axis = df.index

    x_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_X_value,
        name = 'X-Axis Gyro'
    )
    y_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_Y_value,
        name='Y-Axis Gyro'
    )
    z_gyro = go.Scatter(
        x=x_axis,
        y=df.gyr_Z_value,
        name='Z-Axis Gyro'
    )
    x_acc = go.Scatter(
        x=x_axis,
        y=df.acc_X_value,
        name = 'X-Axis Acc'
    )
    y_acc = go.Scatter(
        x=x_axis,
        y=df.acc_Y_value,
        name='Y-Axis Acc'
    )
    z_acc = go.Scatter(
        x=x_axis,
        y=df.acc_Z_value,
        name='Z-Axis Acc'
    )
    
    fig = tools.make_subplots(rows=3, cols=2, shared_xaxes=True,
                              subplot_titles=('X-Axis Gyro', 'Y-Axis Gyro', 
                                              'Z-Axis Gyro','X-Axis Acc', 
                                              'Y-Axis Acc', 'X-Axis Acc'))
    fig.append_trace(x_gyro, 1, 1)
    fig.append_trace(y_gyro, 2, 1)
    fig.append_trace(z_gyro, 3, 1)
    fig.append_trace(x_acc, 1, 2)
    fig.append_trace(y_acc, 2, 2)
    fig.append_trace(z_acc, 3, 2)

    fig['layout'].update(height=800, width=1000, title='Ride Data')
    return iplot(fig)
