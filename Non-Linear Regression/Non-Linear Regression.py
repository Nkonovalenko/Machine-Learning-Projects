import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
#plt.show()

# Buidling a logarithmic model

def sigmoid(x, Beta_1, Beta_2):
	y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
	return y

# Sample sigmoid line that might fit with the data:
beta_1 = 0.10
beta_2 = 1990.0

# Logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# Plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

# Normalize the data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# Optimize parameters for fit line
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# print final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Plot the regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Data with Log Fit')
plt.show()


# Creating train/test split
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
train_y = ydata[msk]

test_x = xdata[~msk]
test_y = ydata[~msk]

# Build model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# Evaluation of model
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-Score: %.2f" % r2_score(y_hat, test_y))
