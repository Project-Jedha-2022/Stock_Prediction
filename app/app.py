import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np

import datetime
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots


### Config
st.set_page_config(
    page_title="Stock prediction",
    page_icon="💸 ",
    layout="wide"
)

DATA_URL = ('https://raw.githubusercontent.com/Project-Jedha-2022/Stock_Prediction/main/datasets/dataset_social_technical_1d.csv')

### App
st.title("Ubisoft stock prediction 🎨", "main")

st.markdown("""
    Utilisation des sources d'information (Financial, Reddit) pour établir une relation avec la valeur du stock. Dans ce cas on va se centrer au tour de l'entreprise Ubisoft.
""")

st.markdown("---")



### Util functions

# Load data
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data['date'] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    return data

# Predict stock value
# à implementer
def predict(future_date):
    """
    Sum up two integers
    Arguments:
        future_date: date
    Returns:
        The predicted value of the stock for the given date
    """
    value_pred = 0
    return value_pred



data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

col1, col2 = st.columns([1, 5])

with col1:
    with st.form("form_dates"):
        start_date = st.date_input("Select a start date :", datetime.date(2018, 1, 1))
        end_date = st.date_input("Select an end date :", datetime.date.today())
        submit = st.form_submit_button("submit")

        if submit:
            st.write("Start date ", start_date)
            st.write("End date ", end_date)
            start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
            mask = (data["date"] >= start_date) & (data["date"] <= end_date)
            data = data[mask]

with col2:
    ### Show data ✅

    st.header("Load and showcase data", "load-and-showcase-data")
    st.markdown("""

        Loading ubisoft dataset. 

    """)

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)    


    ### EDA

    st.header("EDA", "eda")

    features_list = ['SMA_15','Stochastic_15','RSI_15','MACD',
                    'SMA_ratio', 'Stochastic_Ratio', 'RSI_ratio',
                    'SMA15_Volume', 
                    'SMA_Volume_Ratio',
                    'volume',
                    'close',
                    'title_vader_compound',
                    'title_roberta_neg','title_roberta_neu','title_roberta_pos'
                    ]

    target_variable = ['close']
    
    # Correlation Matrix
    st.subheader("Correlation Matrix", "corr")
    
    corr_matrix = data.corr().round(2)
    import plotly.figure_factory as ff
    fig = ff.create_annotated_heatmap(corr_matrix.values,
                                    x = corr_matrix.columns.tolist(),
                                    y = corr_matrix.index.tolist())
    st.plotly_chart(fig, use_container_width=True)

    
    # Histograms
    st.subheader("Histograms", "histograms")
    
    data_histogram = data.drop('date', axis=1) # drop date as it's not used for histogram
    fig = make_subplots(rows=6, cols=3)
    cols = data_histogram.columns
    i = 0
    for col_name in data_histogram.columns:
        row = int(i/3) + 1
        col = i%3 + 1
        fig.add_trace(go.Histogram(x=data[col_name], name=col_name), 
        row=row, col=col)
        i = i + 1
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(width=int(1000),height=int(1000))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment analysis

    st.header("Sentiment analysis", "sentiment_analysis")
    st.markdown("""
        
        Moyenne des sentiments de la communauté en fonction de la date
        
        """)

    col1, col2, col3 = st.columns(3)
    col1.metric()
    col2.metric()
    col3.metric()

    ### Predict

    st.header("Prediction", "predict")
    st.markdown("""

        Prediction de la valeur du stock en fonction de la date. 

    """)

    date_pred = datetime.date.today()
    st.write("Predicted value : ", predict(date_pred))


### Side bar 

st.sidebar.header("Sections")
st.sidebar.markdown("""
    * [Main](#main)
    * [Load and showcase data](#load-and-showcase-data)
    * [EDA](#eda)
    * [Sentiment analysis](#sentiment_analysis)
    * [Predict](#predict)
""")
e = st.sidebar.empty()
e.write("")
st.sidebar.write("Made with 💖 by [Jedha](https://jedha.co)")


### Footer 

empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        🍇
        If you want to learn more, check out [streamlit's documentation](https://docs.streamlit.io/) 📖
    """)

