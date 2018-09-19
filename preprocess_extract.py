import os
import numpy as np 
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt

df= pd.read_csv("fer2013.csv") 

y=df['emotion'].tolist()
data=df['pixels'].tolist()

print(len(data))
print(len(y))
y = list(map(int, y))
x=[]

for i in data:
	j=i.split(" ")
	j = list(map(int, j))
	j=np.array(j)
	j=np.reshape(j,(48,48,1))
	x.append(j)






x=np.array(x)
y=np.array(y)
print('converted to nparray')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


print('saving train test')
np.save('x_k_train',x_train)

np.save('x_k_test',x_test)

np.save('y_k_test',y_test)
np.save('y_K_train',y_train)



