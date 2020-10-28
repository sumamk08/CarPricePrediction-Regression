import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())

# 1. Car ID : Unique id of each observation (Interger) - Drop as not required for prediction
data = data.drop(['car_ID'], axis = 'columns')

data['CompanyName'] = data['CarName'].apply(lambda x : x.split(' ')[0])

data['CompanyName'].replace('maxda','mazda', inplace = True)
data['CompanyName'].replace('porcshce','porsche', inplace = True)
data['CompanyName'].replace('toyouta','toyota', inplace = True)
data['CompanyName'].replace('vokswagen','volkswagen', inplace = True)
data['CompanyName'].replace('vw','volkswagen', inplace = True)

cars_data = data[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'carlength','carwidth']]
print(cars_data)

# Dummy variables

cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['fueltype'], drop_first = True)], axis = 'columns')
cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['aspiration'], drop_first = True)], axis = 'columns')
cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['carbody'], drop_first = True)], axis = 'columns')
cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['drivewheel'], drop_first = True)], axis = 'columns')
cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['enginetype'], drop_first = True)], axis = 'columns')
cars_data = pd.concat([cars_data, pd.get_dummies(cars_data['cylindernumber'], drop_first = True)], axis = 'columns')

cars_data.drop(['fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber'], axis = 'columns', inplace = True)

# Independent & Dependent variable split

X = cars_data.drop(['price'], axis = 'columns')
y = cars_data['price']

# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 0)

# Min Max Scaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns= X_test.columns)

def calculate_vif(X):
    vif_df = pd.DataFrame()
    vif_df['Feature'] =  X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_df['VIF'] = round(vif_df['VIF'], 2)
    vif_df.sort_values(by = 'VIF', ascending = False, inplace=True)
    return vif_df

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_train)
rfe = RFE(lr_model, 10)
rfe.fit(X_train, y_train)
print(rfe.support_)
X_train_final = X_train[X_train.columns[rfe.support_]]
X_test_final = X_test[X_test.columns[rfe.support_]]
lr_model.fit(X_train_final, y_train)
print(calculate_vif(X_train_final))
y_pred = lr_model.predict(X_train_final)
print('R2 score: Training', r2_score(y_train, y_pred))

# carlength  35.82

X_train_final.drop(['carlength'], axis = 'columns', inplace = True)
X_test_final.drop(['carlength'], axis = 'columns', inplace = True)

lr_model.fit(X_train_final, y_train)
y_pred = lr_model.predict(X_train_final)
print('R2 score: Training', r2_score(y_train, y_pred))
y_pred = lr_model.predict(X_test_final)
print('R2 score: Test', r2_score(y_test, y_pred))


print(calculate_vif(X_train_final))

# enginesize  14.73
X_train_final.drop(['enginesize'], axis = 'columns', inplace = True)
X_test_final.drop(['enginesize'], axis = 'columns', inplace = True)

print(calculate_vif(X_train_final))
lr_model.fit(X_train_final, y_train)
y_pred = lr_model.predict(X_train_final)
print('R2 score: Training', r2_score(y_train, y_pred))
y_pred = lr_model.predict(X_test_final)
print('R2 score: Test', r2_score(y_test, y_pred))


y_pred = lr_model.predict(X_test_final)
print(r2_score(y_test, y_pred))