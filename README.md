# Stock-Advisory-Tool
Tool that analyzes the quality of 2 models, ARIMA and LSTM recurrent neural networks, on the intraday historical closing price of a total of 12,250 stocks and ETFs, based on R-squared.


The files "ARIMA-Price" and "LSTM-Price" are the main notebooks of the document The files "ARIMA-Volatility" and "LSTM-Volatility" are additional notebooks. The "Dash function ARIMA_LSTM" file is a notebook where the function was tested and later incorporated into the Dash code for the creation of the visualization application The "listing_status" csv is a list of all the stocks and ETFs available in the Alpha Vantange API (in case the download code for the notebooks fails) The . py files are the files used to launch the application in a terminal. The file that says "LOADED MODEL", uses the saved LSTM model to predict in the application (to gain speed). The file "UNLOADED MODEL", trains and predicts in the application function itself, without needing to load any model (it goes slower but reaches higher scores) The two loaded models .pkl and .h5, are to load the models in the application.


