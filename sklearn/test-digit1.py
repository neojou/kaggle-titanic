
from sklearn import datasets

import matplotlib.pyplot as plt

digits = datasets.load_digits()

for key,value in digits.items() : 
    try:
        print (key, value.shape)
    except:
        print (key)

