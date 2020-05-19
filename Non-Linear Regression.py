import matplotlib.pyplot as plt
import numpy as numpy
import pandas as pd
#%matplotlib inline			# This line is needed for Jupyter Notebook

# Read in the data
df = pd.read_csv("china_gdp.csv")
df.head(10)

# Plot the data set
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title("Data Plot")

plt.show()