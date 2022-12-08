import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#load the model from disk
import joblib
model = joblib.load(r"model.sav")

def preprocess(df, option):

    #This function selects important features, encoding categorical data, handling missing values,feature scaling and splitting the data

    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})

    # Encode binary categorical features
    binary_list = ['SeniorCitizen','Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_list] = df[binary_list].apply(binary_map)

    
    #Drop values based on operational options
    if (option == "Online"):
        columns = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No', 'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes', 'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service', 'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes', 'StreamingMovies_No_internet_service', 'StreamingMovies_Yes', 'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    elif (option == "Batch"):
        pass
        df = df[['SeniorCitizen','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity',
                'OnlineBackup','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                'MonthlyCharges','TotalCharges']]
        columns = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No', 'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes', 'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service', 'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes', 'StreamingMovies_No_internet_service', 'StreamingMovies_Yes', 'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")


    #feature scaling
    sc = MinMaxScaler()
    df['tenure'] = sc.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df

def main():
    #Setting Application title
    st.title('Customer Churn Prediction')

    #Setting Application description
    st.markdown("""
     :dart:  This app helps you Predict Customer Churn in the case of a fictional Telecommunications Company.\n
     Dataset Used : "https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data/Telco-Customer-Churn.csv"
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    add_selectbox = st.sidebar.selectbox(
	"How would you like to Predict?", ("Online", "Batch"))

    if add_selectbox == "Online":
        st.info("Fill in the Details Below to Predict Customer Churn")
        #Based on our optimal features selection
        st.subheader("Demographic Data")
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))


        st.subheader("Payment Data")
        tenure = st.slider('Number of Months the Customer has Stayed with the Company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-Month', 'One Year', 'Two Year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('Amount Charged to the Customer Monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('Total Amount Charged to the Customer',min_value=0, max_value=10000, value=0)

        st.subheader("Services Signed up for")
        mutliplelines = st.selectbox("Does the Customer have Multiple Lines?",('Yes','No','No phone service'))
        phoneservice = st.selectbox('Does the Customer have Phone Service?', ('Yes', 'No'))
        internetservice = st.selectbox("Does the Customer have Internet Service?", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the Customer have Online Security?",('Yes','No','No Internet Service'))
        onlinebackup = st.selectbox("Does the Customer have Online Backup?",('Yes','No','No Internet Service'))
        techsupport = st.selectbox("Does the Customer have Technology Support?", ('Yes','No','No Internet Service'))
        streamingtv = st.selectbox("Does the Customer Stream TV?", ('Yes','No','No Internet Service'))
        streamingmovies = st.selectbox("Does the Customer Stream Movies?", ('Yes','No','No Internet Service'))

        data = {
                'SeniorCitizen': seniorcitizen,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod, 
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Given Input: ')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the Customer will Terminate the Service.')
            else:
                st.success('No, the Customer is Happy with Telco Services!')
        

    else:
        st.subheader("Dataset Upload")
        uploaded_file = st.file_uploader("Choose a File: ")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the Customer will Terminate the Service.', 
                                                    0:'No, the Customer is Happy with Telco Services!'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
