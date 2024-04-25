""""
A program enabling the selection of optimal stations from among the existing networks of measurement stations to determine the spatial average value of the measured environmental parameter in the studied area.
""""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from itertools import combinations

file_name='SST_day.csv'
df=pd. read_csv(file_name)
number_of_stations=2
dependent_variable='All'
independent_variables=df.columns[2:62]
y = df[dependent_variable]

#make a list of all combinations for independent stations 
independent_station_combinations = combinations(independent_variables, number_of_stations)
regr = linear_model.LinearRegression()
best_MSE=np.inf
best_regr=None
best_match=None
for combination in independent_station_combinations:
    x = df[list(combination)]
    # Make a match with sklearn
    regr.fit(x, y)              #Make predictions using the testing set
    y_pred = regr.predict(x)    #Calculate match stats
    MSE=mean_squared_error(y, y_pred)
    if MSE<best_MSE:
        best_MSE=MSE
        best_regr=regr
        best_match=combination
print(best_match)


#Calculate statistics
import statsmodels.api as sm
x = df[list(best_match)]
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(f"RMSE\tR^2\tR^2_adj")
print(f"{np.sqrt(best_MSE):.4f}\t{results.rsquared:.4f}\t{results.rsquared_adj:.4f}")
print(f"Station\t{results.params[0]:.4f}\t{results.bse[0]:.4f}")
for i in range(len(best_match)):
    print(f"{best_match[i]}\t{results.params[i+1]:.4f}\t{results.bse[i+1]:.4f} ")