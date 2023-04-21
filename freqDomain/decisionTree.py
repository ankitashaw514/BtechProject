# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error



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

#print(data)


x= data.iloc [:, : -3] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -1 :] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one
print(x)
print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)





# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# create regressor object
regressor = DecisionTreeRegressor()

# importing numpy library
import numpy as np
from numpy import absolute,mean,std

lr = regressor.fit(X_Train, Y_Train)

# Predicting the test set results


Y_pred = regressor.predict(X_Test) # test the output by changing values

# Calculate Training and Test Accuracy

# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(regressor, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.6f (%.6f)' % (mean(n_scores), std(n_scores)))

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
mae = mean_absolute_percentage_error(Y_Test, Y_pred)


from sklearn.metrics import r2_score
r2 = r2_score(Y_Test, Y_pred)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_Test, Y_pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" % (mae, r2, mse))

#MAE = 0.00033818, R2 = 0.14574500, MSE = 0.00000021

testing_accuracy = lr.score(X_Test,Y_Test)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(X_Train,Y_Train)
print("testing accuracy is: ",traning_accuracy)

total_accuracy = lr.score(x,y)
print("total accuracy is: ",total_accuracy)


'''
real,img,magnitude

MAE = 0.00026729, R2 = 0.36966714, MSE = 0.00000015
testing accuracy is:  0.3696671444679054
testing accuracy is:  0.5128250668458758
total accuracy is:  0.47697857525032883

'''

'''
magnitude

MAE = 0.00012551, R2 = 0.81751144, MSE = 0.00000003
testing accuracy is:  0.8175114362806166
testing accuracy is:  0.8608851113585284
total accuracy is:  0.8500194218171727

'''
