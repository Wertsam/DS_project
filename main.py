import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
pd.set_option('display.max_columns', 20)
df = pd.read_csv("car_stats.csv")

#ax = df['Make'].value_counts().hist()
#ax.set_label("MSRP")
#plt.show()
#df.groupby('Popularity')['Make'].value_counts().plot(kind="bar", title="Популярность каждой марки",figsize=(20,5), color="tomato")
#plt.xlabel('Car Maker')
#plt.ylabel('Popularity')
#plt.show()
fig1 = px.scatter_matrix(
    df, dimensions=['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity', 'MSRP'],
    color="species"
)
fig1.show()
#sns.set(style='whitegrid')
# sns.pairplot(df, plot_kws={'s': 10,'alpha': 0.5})
# plt.suptitle('', y=1.02, fontsize=16)
# plt.show()
df.drop(['Model', 'Market Category', 'Vehicle Size'], axis=1, inplace=True)
# print(df.columns)
# print(df.head(5))
df = df.rename(columns={'Engine Fuel Type': 'Engine_Fuel_Type',
                   "Engine HP": "Engine_HP",
                   "Engine Cylinders": "Engine_Cylinders",
                   "Transmission Type": "Transmission_Type",
                   "Number of Doors": "Number_of_Doors",
                   "Vehicle Style": "Vehicle_Style",
                   "highway MPG": "Highway_MPG",
                   'city mpg': 'City_MPG'})
print(df.columns)
print(df.head(5))
# print(df.isna().sum())
# print(df.describe())
# print(df.duplicated())
# print(df.dtypes)
# print(df["Make"].unique())
# print([df['Engine Cylinders'] == 0])



columns_to_encode = ['Make', 'Engine_Fuel_Type', 'Transmission_Type', 'Driven_Wheels', 'Vehicle_Style']
num_data = df.select_dtypes(exclude='object')

#le = LabelEncoder()

#for column in columns_to_encode:
    #df[column] = ohe.fit(df[column])

ohe = OneHotEncoder(sparse_output=False)


for column in columns_to_encode:

    column_data = df[[column]]


    encoded_data = ohe.fit_transform(column_data)


    new_columns = [f'{column}_{category}' for category in ohe.get_feature_names_out([column])]


    encoded_df = pd.DataFrame(encoded_data, columns=new_columns)


    df = pd.concat([df, encoded_df], axis=1)


df = df.drop(columns=columns_to_encode)
#print(df.head(5))
num_data = df.select_dtypes(exclude='object')
print(num_data.shape)
print(num_data.columns)
#print(num_data.head())

plt.figure(figsize=(10, 10))
#sns.heatmap(num_data.corr(), cmap="RdYlBu_r")
#plt.show()
# cat_data = df.select_dtypes(include='object')
# print(cat_data.shape)
# print(cat_data.head())
