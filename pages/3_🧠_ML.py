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

st.set_page_config(page_title="Predictions", page_icon="🧠")
st.success("# Fraud Detection App 🧠")
st.sidebar.success("Check the type of transaction below")

#Loading data
fraud = pd.read_csv('creditcard.csv')

#Machine Learning
#importing all required libraries for machine learning modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#Drop amounts greater than 2126 and create a new dataframe 'ml_data'. Any amounts above this figure is considered non-fraudulent
ml_data=fraud[fraud['Amount']<=2126]

#Normalizing the time column: using min-max normalization
def min_max(ml_data,col_name):
    ml_data[col_name] = (ml_data[col_name] - ml_data[col_name].min()) / (ml_data[col_name].max() -  ml_data[col_name].min())
    return ml_data

#Scaling the time column
scaled_ml_data = min_max(ml_data, "Time")

#Changing data column to a normal distribution using 'np.log1p'
scaled_ml_data['Time'] = np.log1p(scaled_ml_data['Time'])

#splitting data into independent and dependent variables
#dropping features with an importance of less than 1%
X = scaled_ml_data.drop(['Class','V23','V25'],axis=1)
y = scaled_ml_data['Class']

#Splitting data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.30, random_state=42)

#Initializing and fitting the selected model
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

# get user input
st.success('User Inputs')
col1,col2=st.columns(2)
with col1:
       Time=st.number_input('Time',min_value=0,max_value=1)
       V1=st.number_input('V1',min_value=-57,max_value=3)
       V2=st.number_input('V2',min_value=-74,max_value=23)
       V3=st.number_input('V3',min_value=-35,max_value=11)
       V4=st.number_input('V4',min_value=-7,max_value=18)
       V5=st.number_input('V5',min_value=-25,max_value=36)
       V6=st.number_input('V6',min_value=-28,max_value=18)
       V7=st.number_input('V7',min_value=-45,max_value=23)
       V8=st.number_input('V8',min_value=-75,max_value=22)
       V9=st.number_input('V9',min_value=-15,max_value=17)
       V10=st.number_input('V10',min_value=-26,max_value=25)
       V11=st.number_input('V11',min_value=-6,max_value=14)
       V12=st.number_input('V12',min_value=-20,max_value=8)
       V13=st.number_input('V13',min_value=-7,max_value=6)

with col2:
       V14=st.number_input('V14',min_value=-21,max_value=12)
       V15=st.number_input('V15',min_value=-6,max_value=7)
       V16=st.number_input('V16',min_value=-16,max_value=8)
       V17=st.number_input('V17',min_value=-27,max_value=11)
       V18=st.number_input('V18',min_value=-11,max_value=6)
       V19=st.number_input('V19',min_value=-6,max_value=7)
       V20=st.number_input('V20',min_value=-25,max_value=18)
       V21=st.number_input('V21',min_value=-36,max_value=30)
       V22=st.number_input('V22',min_value=-10,max_value=12)
       V24=st.number_input('V24',min_value=-4,max_value=5)
       V26=st.number_input('V26',min_value=-3,max_value=5)
       V27=st.number_input('V27',min_value=-24,max_value=11)
       V28=st.number_input('V28',min_value=-13,max_value=35)
       Amount=st.number_input('Amount',min_value=0,max_value=2127)
       
#Creating a button to click and predict
st.button("Predict Transaction")
 #Prepare input data for predicction     
pred_new=pd.DataFrame({"Time":[Time],"V1":[V1], "V2":[V2], "V3":[V3], "V4":[V4], "V5":[V5], "V6":[V6], "V7":[V7], "V8":[V8], "V9":[V9], "V10":[V10], "V11":[V11], "V12":[V12], 
                       "V13":[V13], "V14":[V14], "V15":[V15], "V16":[V16], "V17":[V17], "V18":[V18], "V19":[V19], "V20":[V20], "V21":[V21], "V22":[V22], "V24":[V24], "V26":[V26], 
                       "V27":[V27], "V28":[V28], "Amount":[Amount]})

# making predictions using the model
prediction = rf.predict(pred_new)
#Display transaction as either fraudulent or non-fraudulent
if prediction[0] == 1:
        st.sidebar.subheader("Prediction Result")
        st.sidebar.text("The transaction is Fraudulent")
else:
        st.sidebar.subheader("Prediction Result")
        st.sidebar.text("The transaction is Fraudulent")

#Hiding the Streamlit rerun menu from the user
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)