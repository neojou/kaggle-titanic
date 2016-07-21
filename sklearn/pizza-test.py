
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

x = [[6],[8],[10],[14],[18]]
y = [[7],[9],[13],[17.5],[18]]

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('slope: ', regr.coef_)
print('intercept: ', regr.intercept_)

X2 = [[0], [10], [14], [25]]


plt.scatter(x,y, color='black')
plt.plot(X2, regr.predict(X2), color='blue', linewidth=3)
plt.title('pizza price and size')
plt.xlabel('size (inch)')
plt.ylabel('price (USD)')
plt.axis([5,20,5,20])
plt.grid(True)
plt.show()

