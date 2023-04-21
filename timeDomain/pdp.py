# Importing the libraries
from math import dist
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xlsxwriter
import math
data = pd.read_excel('Data.xlsx')

# plot |h(t)|^2 vs timeDelay

def createComplexNumber(row):
   return complex(row[3],row[4])



complexPower = data.apply(createComplexNumber, axis=1)
val = scipy.fft.ifft(complexPower.values,1000)
correctVal = val


book = xlsxwriter.Workbook("data2.xlsx")
sheet = book.add_worksheet()
row = 0
col =0

sheet.write(row,col,"h(t)")
row = row+1
for item in val:
    sheet.write(row,col,item)
    row = row+1

power = []
for item in correctVal:
    power.append(abs(item)*abs(item))


row = 0
col =1
sheet.write(row,col,"power")
row = row+1
for item in power:
    sheet.write(row,col,item)
    row = row+1

col = 0
tempMaxVal = 0
tempMaxIndex = 0

for i in power:
    if i>tempMaxVal:
        tempMaxVal = i
        tempMaxIndex = col
    col= col+1
   



initial = 0
time = []
col = 0
for i in range(0,1000):
    if i>=tempMaxIndex:
        time.append(initial)
        initial = initial+0.4e-10
    


row = 0
col =2
sheet.write(row,col,"timeDelay")
row = row+1
for item in time:
    sheet.write(row,col,item)
    row = row+1

   
col = 0
normalizedPower = []
for i in power:
    
    if col>=tempMaxIndex:
        normalizedPower.append(i/tempMaxVal)
    col = col+1

print(tempMaxIndex) 
powerData = []
for item in normalizedPower:
   powerData.append(20*(math.log(abs(item),10)))



timeData = time

row = 0
col =3
sheet.write(row,col,"normalizedPower")
row = row+1
for item in powerData:
   sheet.write(row,col,item)
   row = row+1

book.close()

print(len(powerData))

print(len(timeData))

figure, (a1) = plt.subplots(1, 1)
a1.plot(timeData,powerData, 'g')
plt.ylabel("Power(db)")
plt.xlabel("Time Delay(ns)")


plt.legend("Power vs Time Delay")
plt.show()



























