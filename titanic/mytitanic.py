
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv('../data/train.csv', header=0)
test_df = pd.read_csv('../data/test.csv', header=0)

cols = ['Name', 'Ticket', 'Cabin']
train_df = train_df.drop(cols, axis=1)
test_df = test_df.drop(cols, axis=1)

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(train_df[col]))

titanic_dummies = pd.concat(dummies, axis=1)
train_df = pd.concat((train_df, titanic_dummies), axis=1)
train_df = train_df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(test_df[col]))

titanic_dummies_test = pd.concat(dummies, axis=1)
test_df = pd.concat((test_df, titanic_dummies_test), axis=1)
test_df = test_df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)



# Fill Age Data
train_df['Age'] = train_df['Age'].interpolate()
test_df['Age'] = test_df['Age'].interpolate()

train_df.info()
test_df.info()



#
# machine learning
#

X = train_df.values
X = np.delete(X, 1, axis=1)
X_tests = test_df.values

y = train_df['Survived'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#use random forests

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)
print rfc.score(X_test, y_test)


from sklearn.preprocessing import Imputer
X_tests = Imputer().fit_transform(X_tests)
y_results = rfc.predict(X_tests)

output = np.column_stack((X_tests[:,0], y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('titanic_results.csv', index=False)



