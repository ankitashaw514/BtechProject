import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('Data.xlsx')

# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(data['Passenger Status'])

# removing the column 'Purchased' from df
# as it is of no use now.
data.drop("Passenger Status", axis=1, inplace=True)

# Appending the array to our dataFrame
# with column name 'Purchased'

data.insert(loc = 2,
          column = 'Passenger Status',
          value = label)
#data["Passenger Status"] = label

data.drop("Magnitude",axis=1,inplace=True)
# printing Dataframe
#print(data)

x= data.iloc [:, : -2] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -2 :] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 1)

lr = knn.fit(xTrain, yTrain)

pred = knn.predict(xTest)

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score,mean_absolute_percentage_error

mae = mean_absolute_percentage_error(yTest,pred)

r2 = r2_score(yTest,pred)

mse = mean_squared_error(yTest,pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" %(mae,r2,mse))

#MAE = 0.00042763, R2 = -0.51745708, MSE = 0.00000038

testing_accuracy = lr.score(xTest,yTest)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(xTrain,yTrain)
print("traning accuracy is: ",traning_accuracy)

'''
magnitude

MAE = 0.00042763, R2 = -0.51745708, MSE = 0.00000038
testing accuracy is:  -0.5174570817768731
traning accuracy is:  -0.3152012275865359

'''