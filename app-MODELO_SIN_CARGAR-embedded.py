
import numpy as np
import pandas as pd
import requests
import csv

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from tensorflow import keras

import dash
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

import time

listing = pd.read_csv("../z.TFM/listing_status.csv")
listing


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.Label("CLOSE PRICE", style={'fontSize':30, 'textAlign':'center'}),
    html.Div([
        dcc.Graph(
            id='stock_graph'
        )
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}
    ),
    
    
    html.Label("SYMBOL:", style={'fontSize':20, 'textAlign':'left'}),
    dcc.Dropdown(
        id='stock_etf_symbol',
        options=[{'label': symbol, 'value': symbol} for symbol in listing.symbol.unique()],
        placeholder="Select a symbol",
        multi=False,
        value=listing.symbol.unique()[0]
    ),
    
    html.Label('TIME FRAME', style={'fontSize':20, 'textAlign':'left'}),
    dcc.RadioItems(
        id="time_frame",
        options=[
            {'label': '60 min', 'value': 60},
            {'label': '30 min', 'value': 30},
            {'label': '15 min', 'value': 15},
            {'label': '5 min', 'value': 5},
        ],
        value=60
           
    ),
    
    html.Label('MODEL', style={'fontSize':20, 'textAlign':'left'}),
    dcc.RadioItems(
        id="model",
        options=[
            {'label': 'ARIMA', 'value': 'ARIMA'},
            {'label': 'LSTM', 'value': 'LSTM'}
        ],
        value="ARIMA"
    )

])

@app.callback(
    dash.dependencies.Output('stock_graph', 'figure'),
    [dash.dependencies.Input('stock_etf_symbol', 'value'),
     dash.dependencies.Input('time_frame', 'value'),
     dash.dependencies.Input('model', 'value')])



def update_graph(symbol, timeframe, model):

    df_filtered = listing[listing['symbol']==symbol]
    

    year = [1,2]
    month = [1,2,3,4,5,6,7,8,9,10,11,12]
    real = pd.DataFrame()

    for y in range(2):
        for m in range(12):
            URL= 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol='+symbol+'&interval='+str(timeframe)+'min&slice=year'+str(year[y])+'month'+str(month[m])+'&apikey=UHV76ERHZOBH4LU8'

            with requests.Session() as s:
                download = s.get(URL)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                my_list = list(cr)

                aux = pd.DataFrame(my_list, columns=("time", "open", "high", "low", "close", "volume"))
                aux.drop(aux.index[0], inplace=True)
                real = real.append(aux, ignore_index=True)

    real.sort_values("time", ascending=True, inplace=True)
    real["time"] = pd.to_datetime(real["time"])
    real["close"] = pd.to_numeric(real["close"])
    real = real[["time", "close"]]
    real.reset_index(drop=True, inplace=True)


    # split train y test
    TEST_SIZE = 0.2

    train = real[:(int(len(real)*(1-TEST_SIZE)))]
    test = real[(int(len(real)*(1-TEST_SIZE))):]

    if model == 'ARIMA':


        # Applying log
        real_log = np.log(real["close"])
        train_log = np.log(train["close"])
        test_log = np.log(test["close"])

        # Training model
        p=1
        d=1
        q=1
        model = ARIMA(train_log, order=(p, d, q))
        fitted_model = model.fit()
        fitted_train = fitted_model.fittedvalues

        # Get Test Set Predictions
        predict_test = fitted_model.predict(test_log.index[0], test_log.index[len(test_log)-1])

         # Reverting from differencing
        fitted_train = fitted_train + train_log.shift(periods=d)
        predict_test = predict_test + test_log.shift(periods=d)

        # Get normal scale
        fitted_train = np.exp(fitted_train)
        predict_test = np.exp(predict_test)

        # Get R2 from Test Set
        real_values = test['close'][d:]
        pred_values = predict_test[d:]
        r2 = r2_score(real_values, pred_values)
        
        
        # Get predictions from whole set
        predictions = fitted_model.predict(real[d:].index[0], real[d:].index[len(real[d:])-1])

        # Reverting from differencing
        predictions = predictions +  real_log[d:].shift(periods=d)

        # Get normal scale
        predictions = np.exp(predictions)
        
        #To calculate naive model R2:
        # Training model
        fitted_train_naive = train.shift(periods=1)

        # Get Test Set Predictions
        predict_test_naive = test.shift(periods=1)

        # Get RMSE from Test Set
        real_values_naive = test['close'][1:]
        pred_values_naive = predict_test_naive['close'][1:]
       

        
        R2_test = r2
        R2_test = np.round(R2_test,3) 
        R2_naive = r2_score(real_values_naive, pred_values_naive)
        R2_naive = np.round(R2_naive, 3)

        
    elif model == 'LSTM':
        train = pd.DataFrame(train['close'])
        test = pd.DataFrame(test['close'])

        # train
        sc = MinMaxScaler(feature_range=(0,1))
        scaled_train = sc.fit_transform(train)

        time_step = 20
        m = len(scaled_train)
        X_train = []
        Y_train = []


        for i in range(time_step,m):
            # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
            X_train.append(scaled_train[i-time_step:i,0])

            # Y: el siguiente dato
            Y_train.append(scaled_train[i,0])

        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


        # test
        scaled_test = sc.transform(test)

        X_test = []
        Y_test = []
        m = len(scaled_test)

        for i in range(time_step, m):
            X_test.append(scaled_test[i-time_step:i,0])
            Y_test.append(scaled_test[i,0])

        X_test, Y_test = np.array(X_test), np.array(Y_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        # model structure
        dim_entrada = (X_train.shape[1],1)
        dim_salida = 1
        na = 8


        model = Sequential()
        model.add(LSTM(na, input_shape=dim_entrada)) 
        model.add(Dropout(rate=0.2))
        model.add(Dense(dim_salida))
        model.compile(loss='mean_squared_error',
                      optimizer="adam",
                      metrics=['mse'])

        # model train
        epochs = 25
        batch_size = 32
        model_trained = model.fit(X_train, Y_train, epochs=epochs,
                                batch_size=batch_size, verbose=2)

         # test and train forescast
        test_close= test.values
        total= pd.concat([train['close'], test['close']],axis=0) 

        test_input = total[len(total)-len(test)-20:].values

        test_input= test_input.reshape(-1,1) 
        scaled_test = sc.transform(test_input)

        X_test= []

        m = len(scaled_test)

        for i in range(time_step,m):
            X_test.append(scaled_test[i-time_step:i,0]) 


        X_test= np.array(X_test)
        X_test= np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


        predicted_value_entrenamiento= model.predict(X_train)
        predicted_value_entrenamiento = sc.inverse_transform(predicted_value_entrenamiento)

        predicted_value= model.predict(X_test)
        predicted_value = sc.inverse_transform(predicted_value)


        pred_train = pd.DataFrame(predicted_value_entrenamiento)
        pred_test = pd.DataFrame(predicted_value)
        predictions = pd.concat([pred_train, pred_test], axis=0, ignore_index=True)
        real = real[time_step:].reset_index()
        
        R2_test= r2_score(test[:], predicted_value)
        R2_test = np.round(R2_test,3)              
        R2_naive = r2_score(test[1:], test[:-1])
        R2_naive = np.round(R2_naive, 3)

    real['pred_close'] = predictions

    fig = px.line(real, x="time", y=["close", "pred_close"], title = 'stock/etf: '+ symbol + ';   R2 test: '+str(R2_test)+';    R2 naive: '+str(R2_naive))
    
    return fig

    



if __name__ == '__main__':
    app.run_server(debug=True)