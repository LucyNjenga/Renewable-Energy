import streamlit as st
import pandas as pd
import numpy as np



df = pd.read_csv("C:\\Users\\user\\Downloads\\WorldBank Renewable Energy Consumption_WorldBank Renewable Energy Consumption (1).csv")
# app title
st.title('RENEWABLE ENERGY PREDICTION MODEL')


#creating a paragraph


st.write('''Renewable energy data encompasses various aspects of energy generated from natural sources that are replenished on a human timescale.''')


 
st.write(df.head(5)) #printing the first 5 rows


#having user slider


num_rows = st.slider("Select the number of rows", min_value = 1, max_value = len(df), value = 5)
st.write("Here are the rows you have selected in the Dataset")
st.write(df.head(num_rows)) #st.write is the print function in python
st.write('The number of rows and columns in the dataset')
st.write(df.shape)
st.write("number of duplicates:", df[df.duplicated()])


#------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Check for duplicates
if st.checkbox('Check for Duplicates'):
    st.write(f'Duplicates in DataFrame: {df.duplicated().sum()}')

#cleaning the outliers
def clean_outliers(column):
  mean = df[column].mean()
  std = df[column].std()
  threshold = 3
  lower_limit = mean - (threshold * std)
  upper_limit = mean + (threshold * std)


  return df[(df[column]>=lower_limit) & (df[column]<=upper_limit)]

#changing the dtype from string to integers
df['Year'] = pd.to_datetime(df['Year'])
df['Year'] = df['Year'].dt.year


columns = ['Year', 'Energy Consump.']
for column in columns:
  new_df = clean_outliers(column)

# Drop Country Code column
new_df = df.drop('Country Code', axis=1)

####################################################################################################
# Encode categorical variables
le = LabelEncoder()

encoded_columns = ['Country Name', 'Income Group', 'Indicator Code', 'Indicator Name', 'Region']
le_dict = {col: LabelEncoder() for col in encoded_columns}

#defining the new dataframe 
for column in encoded_columns:
    le_dict[column].fit(new_df[column])
    new_df[column] = le_dict[column].transform(new_df[column])



#defining our variables
X = new_df.drop('Energy Consump.', axis = 1)
y = new_df['Energy Consump.']


# Train the Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fitting the model
y_train_encoded = le.fit_transform(y_train)
model.fit(X_train, y_train_encoded)

y_test_encoded = le.fit_transform(y_test)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
st.write("R-squared:", r2)


st.sidebar.write("## Enter new data for prediction")


Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)


Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region= st.sidebar.selectbox("Region", le_dict['Region'].classes_)
Income_Group = st.sidebar.selectbox("Income Group",le_dict['Income Group'].classes_)
Year = st.sidebar.number_input("Year")

# Encode user input
encoded_input = [
    le_dict['Country Name'].transform([Country_Name])[0],
   
    le_dict['Indicator Code'].transform([Indicator_Code])[0],
    le_dict['Indicator Name'].transform([Indicator_Name])[0],
    le_dict['Income Group'].transform([Income_Group])[0],
    le_dict['Region'].transform([Region])[0],
    Year,
    
]
 


income_group_map = {
    0: "High income",
    1: "Low income",
    2: "Lower middle income",
    3: "Upper middle income"
}

# Convert the list to a numpy array and reshape
encoded_input = np.array(encoded_input).reshape(1, -1)

# Predict and display the result
if st.sidebar.button('Predict Energy Consumption'):
    prediction = model.predict(encoded_input)[0]
    st.sidebar.write(f'Predicted Energy Consumption: {prediction}')

