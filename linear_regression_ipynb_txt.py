# -*- coding: utf-8 -*-
"""linear_regression.ipynb.txt


"""

# import pandas
import pandas as pd
import numpy as np

from google.colab import files
files.upload()

# read CSV file from the 'data' subdirectory using a relative path
data = pd.read_csv('salary_data.csv')

# display the first 5 rows
data.head(10)

"""Primary object types:

- **DataFrame:** rows and columns (like a spreadsheet)
- **Series:** a single column
"""

# display the last 5 rows
data.tail()

# check the shape of the DataFrame (rows, columns)
data.shape

(100,4)

"""What are the features?
- **TV:** advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
- **Radio:** advertising dollars spent on Radio
- **Newspaper:** advertising dollars spent on Newspaper

What is the response?
- **Sales:** sales of a single product in a given market (in thousands of items)

What else do we know?
- Because the response variable is continuous, this is a **regression** problem.
- There are 200 **observations** (represented by the rows), and each observation is a single market.

## Visualizing data
"""

# Commented out IPython magic to ensure Python compatibility.
# conventional way to import seaborn
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
# %matplotlib inline

data.plot(kind='scatter',x='country_name',y='median_salary',color='red')
data.plot(kind='scatter',x='continent_name',y='median_salary',color='blue')
data.plot(kind='scatter',x='wage_span',y='median_salary',color='green')
data.plot(kind='scatter',x='median_salary',y='median_salary',color='black')
data.plot(kind='scatter',x='average_salary',y='median_salary',color='cyan')
data.plot(kind='scatter',x='wage_span',y='median_salary',color='violet')
data.plot(kind='scatter',x='wage_span',y='median_salary',color='yellow')

"""## Linear regression

**Pros:** fast, no tuning required, highly interpretable, well-understood

**Cons:** unlikely to produce the best predictive accuracy (presumes a linear relationship between the features and response)

### Form of linear regression

$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$

- $y$ is the response
- $\beta_0$ is the intercept
- $\beta_1$ is the coefficient for $x_1$ (the first feature)
- $\beta_n$ is the coefficient for $x_n$ (the nth feature)

In this case:

$y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper$

The $\beta$ values are called the **model coefficients**. These values are "learned" during the model fitting step using the "least squares" criterion. Then, the fitted model can be used to make predictions!

## Preparing X and y using pandas

- scikit-learn expects X (feature matrix) and y (response vector) to be NumPy arrays.
- However, pandas is built on top of NumPy.
- Thus, X can be a pandas DataFrame and y can be a pandas Series!
"""

# create a Python list of feature names
feature_cols = ['median_salary','average_salary',	'lowest_salary',	'highest_salary']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['median_salary','average_salary',	'lowest_salary',	'highest_salary']]

# print the first 5 rows
X.head()

# check the type and shape of X
print(type(X))
print(X.shape)

# select a Series from the DataFrame
lab = ['median_salary']

# equivalent command that works if there are no spaces in the column name
y =  data['median_salary']



# print the first 5 values
y.head()

# check the type and shape of y
print(type(y))
print(y.shape)

"""## Splitting X and y into training and testing sets"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# X_train = X_train[:150]

print(X_test)

"""## Linear regression in scikit-learn"""

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
model = linreg.fit(X_train, y_train)



"""### Making predictions"""

# make predictions on the testing set
y_pred = linreg.predict(X_test)
print(y_pred)

X_test

"""We need an **evaluation metric** in order to compare our predictions with the actual values!"""
