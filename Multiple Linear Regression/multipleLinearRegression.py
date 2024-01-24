# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing Methods from sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Reading the data
data = pd.read_csv("data2.csv")
data.head()
data.shape

print(data.corr())

print(data.describe())

# Identifying dependent and independent variables
X = data[['Weight', 'Volume']]
y = data['CO2']

# Checking for outliers
fig, axs = plt.subplots(2, figsize=(5, 5))
plt1 = sns.boxplot(X['Weight'], ax=axs[0])
plt2 = sns.boxplot(X['Volume'], ax=axs[1])
plt.tight_layout()

# Distribution of the target variable
sns.distplot(y)

# Relationship between CO2 and other variables
sns.pairplot(data, x_vars=['Weight', 'Volume'], y_vars='CO2', height=4, aspect=1, kind='scatter')
plt.show()

# Creating the correlation matrix and represent it as a heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train.shape()
y_test.shape()

reg_model = linear_model.LinearRegression()

# Fitting the Multiple Linear Regression model
reg_model = LinearRegression().fit(X_train, y_train)

# Printing the coefficients
print('Intercept: ',reg_model.intercept_)
# Pairing the feature names with the coefficients
list(zip(X, reg_model.coef_))

# Predicting the Test and Train set results
y_pred = reg_model.predict(X_test)
x_pred = reg_model.predict(X_train)

print("Prediction for test set: {}".format(y_pred))

# Actual value and predicted value comparison
reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
reg_model_diff

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', r2)