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

#data.drop("Magnitude",axis=1, inplace=True)


# printing Dataframe




x= data.iloc [:, : -3] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -3 :-1] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)

print(X_Test)
print(Y_Test)


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# importing numpy library
import numpy as np

lr = regressor.fit(X_Train, Y_Train)

# Predicting the test set results


Y_pred = regressor.predict(X_Test) # test the output by changing values




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error

mae = mean_absolute_percentage_error(Y_Test, Y_pred)



from sklearn.metrics import r2_score
r2 = r2_score(Y_Test, Y_pred)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_Test, Y_pred)

print("MAE = %0.8f, R2 = %0.8f, MSE = %0.8f" % (mae, r2, mse))


#MAE = 0.00033825, R2 = 0.14577672, MSE = 0.00000021

Y_pred_Train = regressor.predict(X_Train) # test the output by changing values

testing_accuracy = lr.score(X_Test,Y_Test)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(X_Train,Y_Train)
print("traning accuracy is: ",traning_accuracy)

total_accuracy = lr.score(x,y)
print("total accuracy is: ",total_accuracy)


'''
real,img,magnitude

MAE = 0.00026733, R2 = 0.36954711, MSE = 0.00000015
testing accuracy is:  0.36954711383273525
traning accuracy is:  0.5120930135699232
total accuracy is:  0.47639259269104395

'''

'''
magnitude

MAE = 0.00012550, R2 = 0.81725909, MSE = 0.00000003
testing accuracy is:  0.8172590871794216
traning accuracy is:  0.8606714411208989
total accuracy is:  0.8497960620037649

'''