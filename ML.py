import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np  # linear algebra
import pandas as pd  # data processing

data = pd.read_csv(r'C:\Users\Sumit\Desktop\insurance.csv')
print("\n THE DATASET IS AS FOLLOWS \n")
data = data.drop('region', axis=1)
print(data)

# creating a label encoder

le = LabelEncoder()
# label encoding for sex
# 0 for females and 1 for males
data['sex'] = le.fit_transform(data['sex'])

# label encoding for smoker
# 0 for smokers and 1 for non smokers
data['smoker'] = le.fit_transform(data['smoker'])


# splitting the dependent and independent variable

x = data.iloc[:, :5]
y = data.iloc[:, 5]

print("\n THE VALUES OF X VARIABLE ARE \n")

print(x)

print("\n THE VALUES OF Y VARIABLE ARE \n")

print(y)

print("\n THE SHAPE OF X & Y VARIABLE ARE \n")

print(x.shape)
print(y.shape)

# splitting the dataset into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=30)

print("\n THE SHAPE OF X & Y TRAIN TEST VARIABLE ARE \n")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# standard scaling

# creating a standard scaler
sc = StandardScaler()

# feeding independents sets into the standard scaler
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Set data
df = pd.DataFrame({
    'group': [i for i in range(0, 1338)],
    'Age': data['age'],
    'Charges': data['charges'],
    'Children': data['children'],
    'BMI': data['bmi']
})


print("##################### MLR ######################")
# importing the model
# Multiple linear regression
model = LinearRegression()


# Fit linear model by passing training dataset
model.fit(x_train, y_train)

# Predicting the target variable for test datset
predictions = model.predict(x_test)

print('THE PREDICTION VALUE IN MULTIPLE LINEAR REGRESSOR IS:\n')
print(predictions)

# plotting the y prediction
plt.scatter(y_test, predictions)
plt.title('Multiple Linear Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# RANDOM FOREST

print("##################### RFR ######################")
# creating the model
model = RandomForestRegressor(n_estimators=40, max_depth=4, n_jobs=-1)

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)
print('THE PREDICTION VALUE OF IN RANDOM FOREST REGRESSOR IS:\n')
print(y_pred)

# plotting the y prediction
plt.scatter(y_test, y_pred)
plt.title('Random Forest Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# feature extraction
print("##################### PCA WITH MLR ######################")

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# importing the model
# Multiple linear regression
model = LinearRegression()


# Fit linear model by passing training dataset
model.fit(x_train, y_train)

# Predicting the target variable for test datset
predictions = model.predict(x_test)

print('THE PREDICTION VALUE OF PCA WITH MULTIPLE LINEAR REGRESSOR IS:\n')
print(predictions)

# plotting the y prediction
plt.scatter(y_test, predictions)
plt.title('PCA with Multiple Linear Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# feature extraction
print("##################### PCA WITH RFR ######################")

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# RANDOM FOREST
# creating the model
model = RandomForestRegressor(n_estimators=40, max_depth=4, n_jobs=-1)

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

print('THE PREDICTION VALUE OF IN PCA WITH RANDOM FOREST REGRESSOR IS:\n')
print(y_pred)

# plotting the y prediction
plt.scatter(y_test, y_pred)
plt.title('PCA with Random Forest Regression')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# Calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("\n r2 score :", r2)
