# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Train.csv')
dataset2 = pd.read_csv('Test.csv')
X_train = dataset.iloc[:, :-1].values
Y_train = dataset.iloc[:, 5].values
X_test = dataset2.iloc[:,:-1].values
Y_test = dataset2.iloc[:,5].values

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=False)
regressor.fit(X_train, Y_train)

print(regressor.coef_)
# Predicting the Test set results
Y_pred = regressor.predict(X_test)

u=((Y_test-Y_pred)**2).sum()
v= ((Y_test - Y_test.mean()) ** 2).sum()
r=(1-(u/v))
print(r)
print(regressor.score(X_test,Y_test))
a=Y_pred
b=Y_test
#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X ,axis = 1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

#Automatic Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:s
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = backwardElimination(X_opt, SL)
regressor_OLS.summary()