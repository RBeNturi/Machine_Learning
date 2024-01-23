# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib as mpl
# mpl.use('TkAgg')

# Reading the data
data = pd.read_csv("score.csv")

# Identifying dependent and independent variables
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Draw a scatter plot
plt.scatter(X, y, color='red')
plt.title('Score Vs Hours of Study')
plt.xlabel('Hours of Study')
plt.ylabel('Score')
plt.show()

# Data cleaning
data.isnull().sum()

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)

# Performance metrics
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('R-squared score: %.2f' % r2_score(y_test, y_pred))
