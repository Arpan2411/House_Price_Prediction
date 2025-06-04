import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
import matplotlib.pyplot as plt

def generate_house_data(samples = 100): # generating synthetic/ artificial house data by using parameter size and price # by default 100 data points are taken by the function
    np.random.seed(50) # every tine while running the function it will give same random numbers
    size = np.random.normal(1400, 50, samples)
    price = size * 50 + np.random.normal(0, 50, samples) # price will be base price ( size * 50) + noise
    return pd.DataFrame({'size':size, 'price': price}) # combines size and price into dataframe

# Importing the required modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
def train_model():
    df = generate_house_data(samples=100)
    X = df[['size']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1) # splitting the dataset into training and test set here test set is 20% of the dataset

    model = LinearRegression() # initializes a linear rtegression model
    model.fit(X_train, y_train)
    return model

def main():
    st.title("House Price Prediction using simple Linear Regression")
    st.write("Put your house size to know the expected price")
    model = train_model()
    size = st.number_input('House Size', min_value=500, max_value=10000, value=1000)
    if st.button('Predict Price'):
        predicted_price = model.predict([[size]])
        st.success(f'Estimated Price : Rs {predicted_price[0]:,.2f}')

        df = generate_house_data()

        # graph using plotly

        # fig = px.scatter(df, x = 'size', y = 'price', title = "Size vs House Price")
        # fig.add_scatter( x= [size], y = [predicted_price[0]],
        #                 mode = 'markers', 
        #                 marker = dict(size= 15, color = 'green'),
        #                 name = 'Prediction')
        # st.plotly_chart(fig)

        # graph using matplotlib

        plt.figure(figsize=(10,6))
        plt.scatter(df['size'], df['price'], label = 'Data Points', alpha = 0.6)
        plt.scatter(size, predicted_price[0], color='red', s=100, label='Prediction')
        plt.title("Size vs House Price")
        plt.xlabel("Size (sq ft)")
        plt.ylabel("Price (Rs)")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)
        


if __name__=='__main__': # run the main function directly based on the condition
    main()
