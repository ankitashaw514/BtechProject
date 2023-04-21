import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)



from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 1)

lr = knn.fit(xTrain, yTrain)

pred = knn.predict(xTest)




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
yPredForsingleData = knn.predict(xForSingleData) # test the output by changing values

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




from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score,mean_absolute_percentage_error

mape = mean_absolute_percentage_error(yTest,pred)

r2 = r2_score(yTest,pred)

mse = mean_squared_error(yTest,pred)

print("MAPE = %0.8f, R2 = %0.8f, MSE = %0.8f" %(mape,r2,mse))

#MAE = 0.00042763, R2 = -0.51745708, MSE = 0.00000038

testing_accuracy = lr.score(xTest,yTest)
print("testing accuracy is: ",testing_accuracy)
traning_accuracy = lr.score(xTrain,yTrain)
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




plotGraph(yTest, pred, "predVsTest for knn")


'''
MAPE = 0.04772891, R2 = 0.53879925, MSE = 251.95288024
testing accuracy is:  0.538799254250069
traning accuracy is:  0.5863941167250933
total accuracy is:  0.574351149493646

'''