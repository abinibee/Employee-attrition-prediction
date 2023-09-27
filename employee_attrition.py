import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression
import seaborn as sns
data_ = pd.read_csv('HR_DATA.csv')

#.......load the model.................
model = joblib.load(open('Employee_attrition.pkl', 'rb'))

#....................Streamlit Development Starts.............

st.markdown("<h1 style = ' color: #34dbeb'>EMPLOYEE ATTRITION PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h5 style = 'top-margin: 0rem; color: #b434eb'>Built By Gbenga Olaosebikan</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)

st.image('staff attrition.png') 

# st.write('Pls Enter your username')
username = st.text_input('Username:')
# password = st.text_input('please enter your password')
# Create a password input field
password = st.text_input('Password', type='password')


# # Get the inputted username from the user
# username = input("Enter a username: ")

# # Check if the username meets certain criteria
# if len(username) < 5:
#     print("Username is too short. Please choose a longer username.")
# elif len(username) > 20:
#     print("Username is too long. Please choose a shorter username.")
# elif not username.isalnum():
#     print("Username contains invalid characters. Please use only letters and numbers.")
# else:
#     print("Username is approved. Welcome, " + username + "!")

#     # Define the predefined password
# correct_password = "mysecretpassword"

# # Prompt the user to input a password
# user_password = input("Enter the password: ")

# # Check if the user input matches the correct password
# if user_password == correct_password:
#     print("Password accepted. Access granted.")
# else:
#     print("Incorrect password. Access denied.")


# Display the entered password (for demonstration purposes)
# st.write('You entered:', password)

st.radio('I am using this prediction for;', ['further academic research work', 'my organization'])
if st.button('Login'): 
   st.success(f"Welcome {username}. Pls enjoy your usage")

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>S Y N O P S I S </h1>", unsafe_allow_html = True)

st.markdown("<p style = 'top margin: 0rem; 'text align: justify; color: #AED2FF'>Attrition of staff, commonly referred to as employee attrition or turnover, is the rate at which employees leave an organization over a specified period. It is an important metric that organizations use to assess their workforce stability and retention efforts. Employee churn is a significant concern for organizations, and managing it effectively is crucial for maintaining a stable and motivated workforce. Understanding the causes, measuring the impact, and implementing retention strategies are essential aspects of HR management. Here are some key features related to staff attrition displayed by the heatmap.</p></h1>", unsafe_allow_html = True) 

# password = ['kjkjk', 'kikik', 'popop', 'pouy']


heat_map = plt.figure(figsize = (14,7))#........create a heatmap plot
correlation_data = data_[['Age', 'MonthlyIncome', 'TotalWorkingYears','OverTime', 'JobSatisfaction', 'YearsInCurrentRole', 'MonthlyRate', 'Attrition']]#.........select data_ for correlation
sns.heatmap(correlation_data.corr(), annot = True, cmap = 'BuPu')

st.write(heat_map)
# data_.drop('user_id', axis = 1, inplace = True)
st.write(data_.sample(5))

st.sidebar.image('HR come up.png', width = 100, caption= f"Welcome {username}", use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)


st.sidebar.write('Pls decide your variable input type')
input_style = st.sidebar.selectbox('Pick Your Preferred input', ['Slider Input', 'Number Input'])

if input_style == 'Slider Input':
    MonthlyIncome = st.sidebar.slider('MonthlyIncome', data_['MonthlyIncome'].min(), data_['MonthlyIncome'].max())
    TotalWorkingYears = st.sidebar.slider('TotalWorkingYears', data_['TotalWorkingYears'].min(), data_['TotalWorkingYears'].max())
    OverTime = st.sidebar.slider('OverTime', data_['OverTime'].min(), data_['OverTime'].max())
    MonthlyRate = st.sidebar.slider('MonthlyRate', data_['MonthlyRate'].min(), data_['MonthlyRate'].max())
    YearsInCurrentRole = st.sidebar.slider('YearsInCurrentRole', data_['YearsInCurrentRole'].min(), data_['YearsInCurrentRole'].max())
    Age = st.sidebar.slider('Age', data_['Age'].min(), data_['Age'].max())
    JobSatisfaction = st.sidebar.slider('JobSatisfaction', data_['JobSatisfaction'].min(), data_['JobSatisfaction'].max())


else:
    MonthlyIncome = st.sidebar.number_input('MonthlyIncome', data_['MonthlyIncome'].min(), data_['MonthlyIncome'].max())
    TotalWorkingYears = st.sidebar.number_input('TotalWorkingYears', data_['TotalWorkingYears'].min(), data_['TotalWorkingYears'].max())
    OverTime = st.sidebar.number_input('OverTime', data_['OverTime'].min(),  data_['OverTime'].max())
    MonthlyRate = st.sidebar.number_input('MonthlyRate', data_['MonthlyRate'].min(),  data_['MonthlyRate'].max())
    YearsInCurrentRole = st.sidebar.number_input('YearsInCurrentRole', data_['YearsInCurrentRole'].min(), data_['YearsInCurrentRole'].max())
    Age = st.sidebar.number_input('Age', data_['Age'].min(), data_['Age'].max())
    JobSatisfaction = st.sidebar.number_input('JobSatisfaction', data_['JobSatisfaction'].min(), data_['JobSatisfaction'].max())

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'MonthlyIncome': MonthlyIncome, 'TotalWorkingYears': TotalWorkingYears, 'OverTime': OverTime, 'MonthlyRate': MonthlyRate, 'YearsInCurrentRole': YearsInCurrentRole, 'Age': Age, 'JobSatisfaction': JobSatisfaction}])

st.write(input_var)

from sklearn.preprocessing import LabelEncoder, StandardScaler
lb = LabelEncoder()
scaler = StandardScaler()

# def transformer(dataframe):
#     # scale the numerical columns
#     for i in dataframe.columns: # ---------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
#             dataframe[[i]] = scaler.fit_transform(dataframe[[i]]) # ------------ Scale all the numericals

#     # label encode the categorical columns
#     for i in dataframe.columns:  # --------------------------------------------- Iterate through the dataframe columns
#         if i in dataframe.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
#             dataframe[i] = lb.fit_transform(dataframe[i]) # -------------------- Label encode selected categorical columns
#     return dataframe

for i in input_var.columns: # ---------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = 'number').columns: # --------- Select only the numerical columns
        input_var[[i]] = scaler.fit_transform(input_var[[i]]) # ------------ Scale all the numericals

for i in input_var.columns:  # --------------------------------------------- Iterate through the dataframe columns
    if i in input_var.select_dtypes(include = ['object', 'category']).columns: #-- Select all categorical columns
        input_var[i] = lb.fit_transform(input_var[i]) # -------------------- Label encode selected categorical columns


st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Attrition is :", prediction)
        if prediction == 0:
            st.toast('PREDICTION SUCCESFUL !!!!')
            st.warning('This individual will not leave the organization')
        if prediction == 1:
            st.toast('PREDICTION SUCCESFUL !!!!')
            st.warning('This individual will leave the organization')    
    else:
        st.write('Pls press the predict button for prediction')


with tab2:
    st.subheader('Model Interpretation')
    
    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"-Provide Support for Employees' Circumstances: Recognize the higher attrition rate among divorced workers and widowed workers and investigate the underlying factors causing this trend.")

    st.markdown(f"-Explore the reasons behind the higher attrition rate among workers aged 60 years and above. Consider implementing targeted retention strategies for older workers, such as flexible work arrangements, mentoring programs, and continued learning and development opportunities. ")

    st.markdown(f"- Foster a culture of inclusivity by providing training and resources that promote diversity awareness and respect within the organization. ")

    st.markdown(f"-  Implement diversity and inclusion programs aimed at promoting equal opportunities and addressing any potential biases or barriers faced by under-represented groups. ")



