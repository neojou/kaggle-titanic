
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)


iris = load_iris()
df = pd.DataFrame(columns=['par','val'])
df['par'] = x.astype(float)
df['val'] = y.astype(float)

y_train = np.asarray(df['val'], dtype="|S6")

model = RandomForestClassifier(n_estimators=1000)
model.fit(df[['par']], y_train)

predict_x = np.array(np.arange(1,100)).T
predict_y = model.predict([predict_x])

plt.scatter(x, y, s=1, label="log(x) with noise")
plt.plot(predict_x, predict_y, c="b", label="random forest")
plt.xlabel("x")
plt.ylabel("f(x) = log(x)")
plt.legend(loc="best")
plt.title("A Basic Log Function")

plt.show()



#model.fit(X, y)

#x_test = np.arange(1, 100)
#y_predicted = model.predict(x_test)

#plt.plt(x_test,y_predicted)
#plt.show()

