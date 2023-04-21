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




# data['complexPower'] = data.apply(createComplexNumber, axis=1)
# complexPower = data['complexPower']

complexPower = data.apply(createComplexNumber, axis=1)

val = scipy.fft.ifft(complexPower.values,18018)

# correctVal = val[::-1]
correctVal = val


book = xlsxwriter.Workbook("dataTry.xlsx")
sheet = book.add_worksheet()

row = 0
col = 0
sheet.write(row,col,"Magnitude")
row = row+1
for item in data['Magnitude']:
   sheet.write(row,col,item)
   row = row+1


row = 0
col =1



sheet.write(row,col,"h(t)")
row = row+1
for item in val:
    sheet.write(row,col,item)
    row = row+1

power = []
for item in correctVal:
    power.append(abs(item)*abs(item))


row = 0
col =2
sheet.write(row,col,"power")
row = row+1
for item in power:
    sheet.write(row,col,item)
    row = row+1  




powerInDb = []


for item in power:
   powerInDb.append(20*(math.log(abs(item),10)))




row = 0
col =3
sheet.write(row,col,"Power(db)")
row = row+1
for item in powerInDb:
   sheet.write(row,col,item)
   row = row+1

row = 0
col = 4
sheet.write(row,col,"Freq")
row = row+1
for item in data['Frequency']:
   sheet.write(row,col,item)
   row = row+1

row = 0
col = 5
sheet.write(row,col,"Distance")
row = row+1
for item in data['Distance']:
   sheet.write(row,col,item)
   row = row+1

row = 0
col = 6
sheet.write(row,col,"Passenger Status")
row = row+1
for item in data['Passenger Status']:
   sheet.write(row,col,item)
   row = row+1




col = 0
tempMaxVal = -10000
tempMaxIndex = 0
maxVal = []
maxIndex = []
r = 1000

for i in power:
    if col<r:
        if i>tempMaxVal:
            tempMaxVal = i
            tempMaxIndex = col
    else:
        maxVal.append(tempMaxVal)
        maxIndex.append(tempMaxIndex)
        tempMaxIndex = 0
        tempMaxVal = -10000
        r = r+1000
    col= col+1

row = 0
col =7

sheet.write(row,col,"LOS")
row = row+1


for item in maxVal:
    for i in range(0,999):
        sheet.write(row,col,item)
        row = row+1

   









book.close()

