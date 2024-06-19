import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

irisset=datasets.load_iris()
X=irisset.data[:50,0:1]
y=irisset.data[:50,1:2]


reg=LinearRegression().fit(X,y)
w=reg.coef_
c=reg.intercept_



xpoints=np.linspace(4,6)
ypoints=w[0]*xpoints+c
plt.plot(xpoints,ypoints,'r-')
plt.scatter(X,y,s=10)
plt.suptitle('Linear Regression IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

ypredict=reg.predict(X)
rmse=np.sqrt(mean_squared_error(y,ypredict))
r2=r2_score(y,ypredict)
print("Train RMSE =",rmse)
print("Train R2 Score =",r2)

