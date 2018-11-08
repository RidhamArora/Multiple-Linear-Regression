
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from decimal import Decimal
# Importing the dataset
dataset = pd.read_csv('Train.csv')
dataset2 = pd.read_csv('Test.csv')
X_train = dataset.iloc[:, :-1].values
Y_train = dataset.iloc[:, 5:].values
X_test = dataset2.iloc[:,:-1].values
Y_test = dataset2.iloc[:,5:].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(X_train)
x_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(Y_train)
y_test = sc_y.transform(Y_test)

"""for i in range(1600):
    for j in range(5):
        X_train[i][j]=Decimal(X_train[i][j])
        X_train[i][j]=round(X_train[i][j],4)

for i in range(400):
    for j in range(5):
        X_test[i][j]=Decimal(X_test[i][j])
        X_test[i][j]=round(X_test[i][j],4)
for i in range(1600):
    Y_train[i][0]=Decimal(Y_train[i][0])
    Y_train[i][0]=round(Y_train[i][0],4)
for i in range(400):
    Y_test[i][0]=Decimal(Y_test[i][0])
    Y_test[i][0]=round(Y_test[i][0],4)"""
#x_train,y_train,x_test,y_test = X_train,Y_train,X_test,Y_test
##Gradient Algorithm
def hypothesis(x_train,theta):
    return theta[0]+(theta[1]*x_train[0])+(theta[2]*x_train[1])+(theta[3]*x_train[2])+(theta[4]*x_train[3])+(theta[5]*x_train[4])
def hypothesis2(x_train,theta):
    return theta[0]+(theta[1]*x_train[:,0])+(theta[2]*x_train[:,1])+(theta[3]*x_train[:,2])+(theta[4]*x_train[:,3])+(theta[5]*x_train[:,4])

def error(X,Y,theta):
    
    m=x_train.shape[0]
    error=0
    
    for i in range(m):
        hx=hypothesis(x_train[i],theta)
        error+=(hx-y_train[i])**2
    
    return error
    
def gradient(x_train,y_train,theta):
    
    grad = np.zeros((6,))
    m=x_train.shape[0]
    
    for i in range(m):
        hx=hypothesis(x_train[i],theta)
        grad[0] += hx-y_train[i]
        grad[1] += (hx-y_train[i])*x_train[i][0]
        grad[2] += (hx-y_train[i])*x_train[i][1]
        grad[3] += (hx-y_train[i])*x_train[i][2]
        grad[4] += (hx-y_train[i])*x_train[i][3]
        grad[5] += (hx-y_train[i])*x_train[i][4]
    return grad
    
###Algorithm
def gradientDescent(x_train,y_train,learning_rate=.0001):
    theta=np.zeros((6,))
    error_list=[]
    itr = 0
    max_itr=100
    e=1000
    theta_list=[]
    while(itr<max_itr):
        grad=gradient(x_train,y_train,theta)
        print(grad)
        e=error(x_train,y_train,theta)
        error_list.append(e)
        theta_list.append((theta[0],theta[1],theta[2],theta[3],theta[4],theta[5]))
        theta[0]=theta[0]-grad[0]*learning_rate
        theta[1]=theta[1]-grad[1]*learning_rate
        theta[2]=theta[2]-grad[2]*learning_rate
        theta[3]=theta[3]-grad[3]*learning_rate
        theta[4]=theta[4]-grad[4]*learning_rate
        theta[5]=theta[5]-grad[5]*learning_rate
        itr+=1
    print(itr)    
    return theta,error_list,theta_list

final_theta,error_list,theta_list=gradientDescent(x_train,y_train)
y_pred=hypothesis2(x_test,final_theta)
Y_pred=sc_y.inverse_transform(y_pred)
Y_pred=np.reshape(Y_pred,(400,1))
u=((Y_test-Y_pred)**2).sum()
v= ((Y_test - Y_test.mean()) ** 2).sum()
r=(1-(u/v))
print(r)
plt.plot(error_list)
plt.show()
y_pred=hypothesis2(x_train,final_theta)
Y_pred=sc_y.inverse_transform(y_pred)
Y_pred=np.reshape(Y_pred,(1600,1))
u=((Y_train-Y_pred)**2).sum()
v= ((Y_train - Y_train.mean()) ** 2).sum()
rr=(1-(u/v))
print(rr)
for i in range(5):
    plt.scatter(x_test[:,i],y_pred,color='blue')
    plt.scatter(x_test[:,i],y_test,color='Orange')
    plt.show()