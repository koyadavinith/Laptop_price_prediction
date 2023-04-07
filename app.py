import pickle
import streamlit as st
import sklearn
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the data from the pickle file
with open('encode_dict', 'rb') as f:
    encode_dict = pickle.load(f)

with open('rfmodel.pkl', 'rb') as f:
    rf = pickle.load(f)


# Define the options for each feature
ram_options = list(encode_dict['ram'].keys())
os_options = list(encode_dict['os'].keys())
graphics_options = list(encode_dict['graphics'].keys())
processor_options = list(encode_dict['processor'].keys())
memory_options = list(encode_dict['memory'].keys())

# Define the title of the app
st.title('Laptop Price Predictor')

# Define the select boxes for each feature
ram = st.selectbox('Ram', ram_options)
os = st.selectbox('Operating System', os_options)
graphics = st.selectbox('Graphic Card Size', graphics_options)
processor = st.selectbox('Processor', processor_options)
memory = st.selectbox('Memory', memory_options)

# Define the submit button
submit = st.button('Submit')

# Define the function to predict the price
def predict_price(ram, os, graphics, processor, memory):
    
    arr_res =  rf.predict(np.array([encode_dict['ram'][ram],encode_dict['os'][os],encode_dict['graphics'][graphics],\
                         encode_dict['processor'][processor],encode_dict['memory'][memory]]).reshape(1,-1))
    return int(np.round(arr_res[0]))

# When the submit button is clicked, call the predict_price() function
if submit:
    # Call the predict_price() function
    predicted_price = predict_price(ram, os, graphics, processor, memory)
    
    # Format the predicted price with comma separator and Indian Rupee symbol
    formatted_price = '₹ {:,}'.format(predicted_price)
    
    # Create a container to display the predicted price
    result_container = st.container()
    
    # Add a header to the container
    with result_container:
        st.subheader('Predicted Price is')
    
    st.markdown(f"<h1 style='text-align: center; color: green;'>₹{predicted_price:,}</h1>", unsafe_allow_html=True)

