#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt

import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline

# Read in the data
df = pd.read_csv("FuelConsumption.csv")

# Loop at the data set
df.head()

# Set up categories
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Plot emission values with respect to Engine size:
plt.title('All Data')
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# Train/Test Split
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data distribution
plt.title('Train Data')
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()