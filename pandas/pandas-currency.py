
import pandas as pd

url = 'http://www.stockq.org/market/currency.php'
table = pd.read_html(url)[4]
table = table.drop(table.index[[0,1,2,3,4,5,6,34]], axis=0)
table = table.drop(table.columns[5:149], axis=1)

print table

