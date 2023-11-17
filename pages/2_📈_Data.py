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

st.set_page_config(
    page_title="Exploring the data",
    page_icon="📖",
)
st.write("## Exploring data🚀")
st.sidebar.success("Getting facts")

#Loading data
fraud = pd.read_csv('creditcard.csv')

#Getting user input on how many rows to display
rows = st.slider('Slide to see part of data',0,15,3)

#Display data as a dataframe
st.dataframe(fraud.head(rows))

st.download_button("Download data",
                   fraud.to_csv(index=False),
                   file_name="Credit_Card_data.csv")

#Removing duplicates
fraud.drop_duplicates(inplace=True)

#Time feature distribution
st.subheader("Time distribution in seconds")
plt.figure(figsize=(10,4))
plt.title('Time distribution in seconds')
sns.distplot(fraud['Time'], color='Green')
st.pyplot()
st.markdown("""
* As it can be observed, the time feature has a two ***peaks*** indicating a ***bimodal distribution***. This suggests that there are two periods 
when the transactions are more frequent. However, the time feature does not clearly indicate the time of the day when the transactions took 
place but rather the timing.This distribution suggests the usefulness of time feature in credit card fraud detection
""")


#Distribution of amounts
st.subheader("Distribution of Amounts")
plt.figure(figsize=(8,4))
plt.title('Distribution of Amounts')
sns.distplot(fraud['Amount'], color='Green')
st.pyplot()
st.markdown('''
* The amounts distribution is higly skewed to the right with a very long tail. This suggests that most transactions involve less amounts and very 
            few transactions involve large amounts as per the density. This futher indicates the possibility of outliers the amounts feature. This 
            suggests that when building a fraud detection model, it will be necessary to handle outliers in the amounts feature such as by using 
            log transformation or robust statistical methods


''')



#Group the amounts in groups of 100 and plot a graph
cohorts=fraud.groupby(pd.cut(fraud['Amount'], np.arange(0,2500,100))).count()
ax=cohorts['Amount'].plot(kind='bar', figsize=(15,6), title="Amounts grouped in 100s")
ax.bar_label(ax.containers[0], fontsize=10);
st.pyplot()

#Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)