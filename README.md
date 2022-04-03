# Stock-Advisory-Tool
Tool that analyzes the quality of 2 models, ARIMA and LSTM recurrent neural networks, on the intraday historical closing price of a total of 12,250 stocks and ETFs, based on R-squared.


Los archivos "ARIMA-Precio" y "LSTM-Precio" son los notebooks principales del documento
Los archivos "ARIMA-Volatilidad" y "LSTM-Volatilidad" son notebooks adicionales
El archivo "función de Dash ARIMA_LSTM" es un notebook dónde se probó la función que posteriormente se incorpora en el código de Dash para la creación de la aplicación de visualización
El csv "listing_status" es un listado con todos los stocks y ETFs disponibles en la API Alpha Vantange (por si falla el códgio de descarga de los notebooks)
Los archivos .py son los archivos empleados para lanzar la aplicación en una terminal. El archivo que pone "MODELO CARGADO", utiliza el modelo LSTM guardado para predecir en la aplicación (para ganar velocidad). El archivo "MODELO SIN CARGAR", entrena y predice en la propia función de la aplicación, sin necesitar cargar ningún modelo (va más lento pero llega a scores mayores)
Los dos modelos cargados .pkl y .h5, son para cargar los modelos en la aplicación
