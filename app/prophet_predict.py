import prophet
from prophet import Prophet

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot as plt
import yfinance as yf

# from fbprophet.plot import plot_cross_validation_metric
# from prophet.plot import add_changepoints_to_plot
from pickle import dump
from pickle import load

class my_prophet:
    def __init__(self, period_length):
        self.period_len = period_length
        self.model2 = load(open('prophet_model.pkl', 'rb'))
        self.future = self.model2.make_future_dataframe(periods = self.period_len)
        df = pd.read_csv('dataset_social_technical_1d.csv')
        print(df.head())

        self.future[['close', 'volume', 'SMA_15', 'SMA_ratio', 'SMA15_Volume',
           'SMA_Volume_Ratio', 'Stochastic_15', 'Stochastic_Ratio', 'RSI_15',
           'RSI_ratio', 'MACD', 'price_pct_variation', 'title_vader_compound',
           'title_roberta_neg', 'title_roberta_neu', 'title_roberta_pos']] = 0
        self.forecast = self.model2.predict(self.future)


    def prophet_predict(self, the_date):
        mask = (self.forecast['ds'] == the_date)
        return(self.forecast['yhat'][mask].iloc[0])

if __name__ == "__main__":
    prediction_period_length = 100
    prophet_model = my_prophet(prediction_period_length)
    the_date = '2022-11-13'
    stock_price = prophet_model.prophet_predict(the_date)
    print("value predicted for {} = {}".format(the_date, stock_price))
