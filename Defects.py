import pandas as pd
import pandas as pd  
import numpy as np
from numpy import cov
import seaborn as sns
import matplotlib.pyplot as plt  
#import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

dataset=pd.read_csv('C:\data backup\sdalavi\siddhesh1\Python\dataset.csv')

#dataset.describe()

#correlation=dataset.corr()

#f, ax = plt.subplots(figsize =(9, 8)) 
#print(sns.heatmap(correlation, ax = ax, cmap ="YlGnBu", linewidths = 0.1))

#print(correlation)

df=pd.DataFrame(dataset)

#print(df)

X=df[['No of TCs Executed','No of Defects - SIT']]
y=df[['No of ACTUAL Defects - UAT']]

#covariance = cov(X)
#print(covariance)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state=0)

print("X_train is:",X_train)
print(X_train.describe())
print("X_test is:",X_test)


            #---Data scaling as follows---
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#print("printX scalar train is :",X_train)
#X_test = scaler.transform(X_test)
#print(X_test)



            #---build the model---
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
#print("reg is:",reg.fit)

#To retrieve the intercept:
print("Y intercept is:",reg.intercept_)

#For retrieving the slope:
print("Coeficients of Xs are :",reg.coef_)


y_pred = reg.predict(X_test)

print("predicted UAT defects for Testing Dataset:",y_pred)
#print("X is :", abc)
#y_test_pred=reg.linear.predict(y_test)

y_test=np.array(y_test)
y_pred=np.array(y_pred)
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print("Predicted values Vs Actual values for test data\n", df1)



df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(reg.score(X,y))
