# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
# importing pandas as pd
import pandas as pd



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
data.drop("Magnitude",axis=1, inplace=True)
# printing Dataframe

print(data)


x= data.iloc [:, : -2] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -2 :] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)




# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.linear_model import LinearRegression

# create regressor object
regressor = LinearRegression()

# importing numpy library
import numpy as np

lr = regressor.fit(X_Train, Y_Train)

# Predicting the test set results


Y_pred = regressor.predict(X_Test) # test the output by changing values

# Calculate Training and Test Accuracy



from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(Y_Test, Y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(Y_Test, Y_pred)


from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
mse = mean_absolute_percentage_error(Y_Test, Y_pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" % (mae, r2, mse))

#MAE = 0.00037599, R2 = -0.00125946, MSE = 0.00000025

testing_accuracy = lr.score(X_Test,Y_Test)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(X_Train,Y_Train)
print("traning accuracy is: ",traning_accuracy)

'''
MAE = 0.00037599, R2 = -0.00125946, MSE = 0.00000025
testing accuracy is:  -0.0012594573562312306
traning accuracy is:  0.0001527629309925893

'''