# write your code here
import numpy as np
import pandas as pd

data = {'x0': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'x1': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
        'x2': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
        'y': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]}

df = pd.DataFrame(data)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

'''
Stage 1/4: Fit a linear model
class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = 0
        self.intercept = 0

    # beta = (X_transpose*X)inverse*X_transpose*y
    beta = np.array([0, 0])
    weights = {'Intercept': 0, 'Coefficient': 0}

    def fit(self, X, y):
        self.beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.weights['Intercept'] = self.beta[0]
        self.weights['Coefficient'] = self.beta[1]
        return self.weights
'''


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.y_pred = None
        self.fit_intercept = fit_intercept
        self.coefficient = 0
        self.intercept = 0

    def fit(self, X, y):
        if self.fit_intercept:
            X.insert(0, 'ones', 1)
            X = X.values

        else:
            X = X.values

        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y)).reshape(-1, 1)
        return self.weights

    def predict(self, X):
        if self.fit_intercept:
            pass
        else:
            X = X.iloc[:, 1:]

        self.y_pred = np.dot(X.values, self.weights).reshape(-1, )
        return self.y_pred

    def r2_score(self, y, yhat):
        y = y.values
        self.r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return self.r2

    def rmse(self, y, yhat):
        self.mse = np.mean((y.values - yhat) ** 2)
        return np.sqrt(self.mse)

'''
Stage 2
regCustom = CustomLinearRegression(fit_intercept=False)
regCustom.fit(X, y)
y_pred = regCustom.predict(X)
print(y_pred)

Stage 3/4: Implement metrics
Print the intercept, coefficient, RMSE, and R2 values as a Python dictionary.
regCustom = CustomLinearRegression(fit_intercept=True)
fit = regCustom.fit(X, y)
intercept = fit[0]
coefficient1 = fit[1]
coefficient2 = fit[2]
y_pred = regCustom.predict(X)
rmse = regCustom.rmse(y, y_pred)
r2 = regCustom.r2_score(y, y_pred)
output = {'Intercept': intercept, 'Coefficient': np.array([coefficient1, coefficient2]), 'R2': r2, 'RMSE': rmse}
#print(output)
'''

# Stage 4/4: Compare with the Scikit-Learn Regression algorithm

data = {'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
        'f2': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3],
        'f3': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
        'y': [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
        }

df = pd.DataFrame(data)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

modelSci = LinearRegression(fit_intercept=True)
modelSci.fit(X, y)
interceptSci = modelSci.intercept_
coeffSci = modelSci.coef_
y_predSci = modelSci.predict(X)
r2Sci = r2_score(y, y_predSci)
rmseSci = np.sqrt(mean_squared_error(y, y_predSci))

modelCust = CustomLinearRegression(fit_intercept=True)
fit = modelCust.fit(X, y)
interceptCust = fit[0]
coeffCust = np.zeros(X.shape[1]-1)
for i in range(X.shape[1]-1):
    coeffCust[i] = fit[i+1]

y_predCust = modelCust.predict(X)

r2Cust = modelCust.r2_score(y, y_predCust)
rmseCust = modelCust.rmse(y, y_predCust)

output = {'Intercept': interceptSci - interceptCust, 'Coefficient': coeffSci - coeffCust, 'R2': r2Sci - r2Cust, 'RMSE': rmseSci - rmseCust}
print(output)
