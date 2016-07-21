
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
#create data, y = log(x) + random
#

x = np.linspace(1, 100, 1000).reshape(-1,1) 
lr_y = np.log(x) + np.random.normal(0, 0.3, size=x.shape)
y = np.asarray(lr_y, dtype='|S6')
y = np.ravel(y)


#linear model
lr = linear_model.LinearRegression()
lr.fit(x, lr_y)


#non-linear model

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x, y)

x1 = np.linspace(1, 100, 10).reshape(-1,1)
result=np.ravel(np.log(x1))
result=np.array(result, dtype=float)

lr_result=lr.predict(x1)
lr_result=np.array(lr_result, dtype=float)
lr_diff = lr_result - result
print 'lr_var=',lr_diff.var()


rfc_result=rfc.predict(x1)
rfc_result=np.array(rfc_result, dtype=float)
rfc_diff = rfc_result - result
print 'rfc_var=',rfc_diff.var()

plt.scatter(x, y, s=1, label="log(x) with noise")
plt.plot(x1, result, c="r", label="log")
plt.plot(x1, lr_result, c="b", label="linear")
plt.plot(x1, rfc_result, c="g", label="random forest")

plt.xlabel("x")
plt.ylabel("f(x) = log(x)")
plt.legend(loc="best")
plt.title("A Basic Log Function")

plt.show()

