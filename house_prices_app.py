import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import folium as fl
from streamlit_folium import st_folium


st.write("""
# House Price Prediction App
This app predicts the average property price in a given area of california!
Data obtained from the [hands on machine learning](https://github.com/ageron/handson-ml3) in R by Allison Horst.
""")

tab1, tab2 = st.tabs(["Predict", "Model Info"])

with tab1:
   st.header("Predict")
   st.write("""
    Enter some information about the house type below:
    """)

   # user inputs
   ocean_proximity = st.selectbox('Ocean Proximity', ('<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'))
   median_income = st.slider('Median Income', 0, 20, 10)
   households = st.slider('Number of Households in Area', 1, 6500, 100)
   bedrooms_per_household = st.slider('Bedrooms per Household', 1, 35, 2)
   rooms_per_household = st.slider('Number of Rooms per Household', 1, 150, 5)
   population_per_household = st.slider('Population per Household', 1, 40000, 2000)
   housing_median_age = st.slider('Median Household Age', 1, 100, 30)

   def get_pos(lat,lng):
    return lat,lng

   m = fl.Map()

   m.add_child(fl.LatLngPopup())

   map = st_folium(m, height=350, width=700)

   latitude, longitude = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

   if latitude is not None:
      st.write(latitude)

   # fixed inputs
   total_rooms = households
   total_bedrooms = households
   population = households

   input_data = {
      'longitude': longitude,
      'latitude': latitude,
      'housing_median_age': housing_median_age,
      'total_rooms': total_rooms,
      'total_bedrooms': total_bedrooms,
      'population': population,
      'households': households,
      'median_income': median_income,
      'ocean_proximity': ocean_proximity,
      'bedrooms_per_household': bedrooms_per_household,
      'population_per_household': population_per_household,
      'rooms_per_household': rooms_per_household
   }
   
   input_df = pd.DataFrame(input_data, index=[0])

   full_housing_data = pd.read_csv('housing.csv')
   full_housing_data = full_housing_data.drop(columns=['median_house_value'])
   df = pd.concat([input_df, full_housing_data],axis=0)
   
   features = list(df.columns)

   # encode categorical variables
   data_cat = df.select_dtypes(exclude=[np.number])
   cat_encoder = OneHotEncoder(sparse=False)
   data_cat_1hot = cat_encoder.fit_transform(data_cat)

   features = features + cat_encoder.categories_[0].tolist()
   features.remove("ocean_proximity")

   # scaling
   scaler = StandardScaler()
   data_num = df.select_dtypes(include=[np.number])
   data_scaled = scaler.fit_transform(data_num)

   # concatenating data 
   transformed_input = np.hstack([data_scaled, data_cat_1hot])

   # select input data
   transformed_input = transformed_input[:1]
   
   # Reads in saved model
   load_clf = pickle.load(open('house_model_xgb.pkl', 'rb'))

   # Apply model to make predictions
   prediction = load_clf.predict(transformed_input)
   
   st.subheader('Prediction')
     
   st.write(prediction)

with tab2:
   st.header("Model Info")
   st.write("""
    The model used in this app was trained on the california housing dataset provided as part of the textbook "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow". This project was an adaptation of chapter 2 of this textbook by creating this streamlit app as well as test other models outside what was tested in the chapter.  
    The machine learning algorithm used to make predictions in this app is random forest regression. I also tested many other regression models as well as combining them in an ensemble. However, the Random Forests method returned the best scores.
    
    """)

