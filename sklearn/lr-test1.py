
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model

x = np.linspace(0,10,10).reshape(-1,1)
print "x=", x
y = 2*x+1 + np.random.normal(size=x.shape)
print "y=",y

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('slope: ', regr.coef_)
print('intercept: ', regr.intercept_)


plt.scatter(x,y, color='black')
plt.plot(x, regr.predict(x), color='blue')
plt.title('y=2x+1')
plt.show()

