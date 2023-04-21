# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
# importing pandas as pd
import pandas as pd



data = pd.read_excel('data1.xlsx')



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

data.insert(loc = 6,
          column = 'Passenger Status',
          value = label)
#data["Passenger Status"] = label
#data.drop("Magnitude",axis=1, inplace=True)
# printing Dataframe

#print(data)


x= data.iloc [:, -4: ] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -5 :-4] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one
print(x)
print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.25, random_state = 0)




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



# for distance 149

singleData = pd.read_excel('dataFor149.xlsx')

# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(singleData['Passenger Status'])


# removing the column 'Purchased' from df
# as it is of no use now.
singleData.drop("Passenger Status", axis=1, inplace=True)

# Appending the array to our dataFrame
# with column name 'Purchased'

singleData.insert(loc = 6,
          column = 'Passenger Status',
          value = label)
        

xForSingleData = singleData.iloc [:, -4: ]
yForSingleData = singleData.iloc[:,-5:-4]
yForSingleDataVal = np.array(yForSingleData)
yPredForsingleData = regressor.predict(xForSingleData) # test the output by changing values

powerForSingleData = yForSingleData[::-1]






# for predicted values

col = 0
tempMaxVal = -10000
tempMaxIndex = 0

for i in yPredForsingleData:
    if i>tempMaxVal:
        tempMaxVal = i
        tempMaxIndex = col
    col= col+1

col = 0
powerForSingleDataPred = []

for i in yPredForsingleData:
    if col>=tempMaxIndex:
        powerForSingleDataPred.append(i/tempMaxVal)
    col= col+1


length = len(powerForSingleDataPred)
initial = 0
time = []
for i in range(0,length):
    time.append(initial)
    initial = initial+0.4e-10


powerSingleDataPred = powerForSingleDataPred[::-1]






figure, (a1) = plt.subplots(1, 1)
a1.plot(time,powerSingleDataPred, 'g')
plt.ylabel("Power(db)")
plt.xlabel("Time Delay(ns)")


plt.legend("Power vs Time Delay")
plt.show()





from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error

mape = mean_absolute_percentage_error(Y_Test, Y_pred)



from sklearn.metrics import r2_score
r2 = r2_score(Y_Test, Y_pred)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_Test, Y_pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" % (mape, r2, mse))


#MAE = 0.00033825, R2 = 0.14577672, MSE = 0.00000021

Y_pred_Train = regressor.predict(X_Train) # test the output by changing values

testing_accuracy = lr.score(X_Test,Y_Test)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(X_Train,Y_Train)
print("traning accuracy is: ",traning_accuracy)

total_accuracy = lr.score(x,y)
print("total accuracy is: ",total_accuracy)





def plotGraph(y_test,y_pred,regressorName):
    
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.legend(['yTest','yPredicted'])
   

    plt.title(regressorName)
    plt.show()
    return




plotGraph(Y_Test, Y_pred, "predVsTest for RandomForest")


'''
MAPE = 0.03675318, R2 = 0.73515032, MSE = 144.68675408
testing accuracy is:  0.7351503232820509
traning accuracy is:  0.7934829188890944
total accuracy is:  0.7787148340855932

'''