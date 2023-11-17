import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import warnings

#Ignore Warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

#Load the credit card dataset
fraud = pd.read_csv('creditcard.csv')

st.set_page_config(
    page_title="Hello",
    page_icon="🏡",)

st.success("Credit Card Fraud detection ML web App 🏧")
#Adding the sidebar
st.sidebar.success("Select a demo above.")

#Using beta columns
col1,col2 = st.columns(2)
with col1:
     st.success('About the data')
     st.markdown("""
Credit card fraud is a major concern for financial institutions. Fraudsters use sophisticated and 
dubious means to steal or access credit card information which they in turn use to make fraudulent transactions.
            
In this project, I will be working to come up with a model that predicts unauthorized/fraudulent transactions. The dataset
which I acquired from Kaggle has transactions made by card holders from Europe. The data has 283253 non-fraudulent(0) and 473 fraudulent(1) transactions.
The data also has 31 columns: Time, V1 through V28, Amount and Class(our target variable) which indicates whether  transaction is fraudulent or not.
V1 through V28 due to confidentiality are all numerical as a result Principal Component Analysis.

""")
    
with col2:
    st.success('About the project')
    st.markdown("""
I started by cleaning the data: checking null values and duplicates and removing them. I did an Exploratory Data Analysis(EDA) buy using charts
and boxplots to understand the data more. I did data preprocessing by dropping all the amounts that do not have any fraudulent transaction in them to avoid overfitting.
I also did data normalization and scaling for easy and good modeling.
            
Finally, I built Machine Learning models using several classification algorithms. The idea was to come up with the best performing 
algorithm which was finally selected for modelling and the deployment of the ML web app.          

""")

#Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)