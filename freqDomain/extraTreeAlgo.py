import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


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

# printing Dataframe

#print(data)

# Separating the dependent and independent variables
x= data.iloc [:, : -3] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -3 :] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Building the model
extra_tree_forest = ExtraTreesRegressor(n_estimators = 100,
										criterion ='squared_error', max_features = 2)

# Training the model
lr = extra_tree_forest.fit(X_Train, Y_Train)

Y_pred = extra_tree_forest.predict(X_Test) # test the output by changing values
print(X_Test)
print(Y_Test)





from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error

mae = mean_absolute_percentage_error(Y_Test, Y_pred)



from sklearn.metrics import r2_score
r2 = r2_score(Y_Test, Y_pred)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_Test, Y_pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" % (mae, r2, mse))
#MAE = 0.00027797, R2 = 0.21954345, MSE = 0.00000019

testing_accuracy = lr.score(X_Test,Y_Test)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(X_Train,Y_Train)
print("traning accuracy is: ",traning_accuracy)
total_accuracy = lr.score(x,y)
print("total accuracy is: ",total_accuracy)


'''
only magnitude

MAE = 0.00012551, R2 = 0.81751144, MSE = 0.00000003
testing accuracy is:  0.8175114362806166
traning accuracy is:  0.8608851113585284
total accuracy is:  0.8500194218171727

'''




'''
real,img,magnitude

MAE = 0.00026729, R2 = 0.36966714, MSE = 0.00000015
testing accuracy is:  0.36966714446790555
traning accuracy is:  0.5128250668458757
total accuracy is:  0.47697857525032883

'''

