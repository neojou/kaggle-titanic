
import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

def do_list(mylist, myfunc):
    R = []
    for item in mylist:
       R.append(myfunc(item))
    return R

def mysplit_get_first(item):
    return item.split(',')[0]


def convert_name(df):
     name1_list = do_list(df['Name'].tolist(), mysplit_get_first)
     df['Name1']=name1_list

def fill_age(df):
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Gender ==i) &
                     (df.Pclass == j+1), \
                     'AgeFill'] = median_ages[i,j]

#
# main
#

train_df = pd.read_csv('../data/train.csv', header=0)
test_df = pd.read_csv('../data/test.csv', header=0)



# Name convertion
convert_name(train_df)
convert_name(test_df)

total_name_list = train_df['Name1'].tolist() + test_df['Name1'].tolist()
le = LabelEncoder()
le.fit(np.array(total_name_list))
train_df['Name2'] = le.transform(train_df['Name1'])
test_df['Name2'] = le.transform(test_df['Name1'])

#print train_df[['Name1', 'Name2']]

# Sex

train_df['Gender'] = train_df['Sex'].map( {'female':0, 'male':1} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female':0, 'male':1} ).astype(int)


# Ticket
total_ticket_list = train_df['Ticket'].tolist() + test_df['Ticket'].tolist()
le = LabelEncoder()
le.fit(np.array(total_ticket_list))
train_df['Ticket1'] = le.transform(train_df['Ticket'])
test_df['Ticket1'] = le.transform(test_df['Ticket'])

# Age 
median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train_df[(train_df['Gender'] == i) & \
                                    (train_df['Pclass'] == j+1)]['Age'].dropna().median()

#print median_ages

fill_age(train_df)
fill_age(test_df)

#Fare
test_df['Fare'] = test_df['Fare'].interpolate()

# Cabin
#print train_df['Cabin']

# FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

#Age*Class
train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass

#delete
cols = ['Name', 'Name1', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Age', 'Cabin', 'Embarked']
train_df = train_df.drop(cols, axis=1)    
test_df = test_df.drop(cols, axis=1)    


print train_df.info()
print test_df.info()

#
# machine learning
#
X = train_df.values
X = np.delete(X, 1, axis=1)
X_tests = test_df.values

y = train_df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)


#use random forests
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)
print rfc.score(X_test, y_test)


#from sklearn.preprocessing import Imputer
#X_tests = Imputer().fit_transform(X_tests)
y_results = rfc.predict(X_tests)

output = np.column_stack((X_tests[:,0], y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('titanic_results-1.csv', index=False)



