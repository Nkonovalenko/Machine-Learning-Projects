import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#%matplotlib inline       # This line is needed for Jupyter Notebook

# Read in the data
df = pd.read_csv("FuelConsumption.csv",error_bad_lines=False)

# look at dataset
df.head()

# Set up categories
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Plot Emission values with respect to Engine Size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
#plt.show()

# Creating train/test split
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Test data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Set up Polynomial Preprocessing, 2nd degree Polynomial
poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)
#print(train_x_poly)

# Solve using linear Regression
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

# Print the coefficients
print("Coefficients: ", clf.coef_)
print("Intercept: ", clf.intercept_)

# Plot
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*(XX**2)
plt.plot(XX, yy, '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

