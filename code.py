import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(df_sales['date'],df_sales['sales'])
# plt.plot(y,sales)
plt.show()
import warnings
warnings.filterwarnings("ignore")
pip install plotly
pip install cufflinks

import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

pyoff.init_notebook_mode()

df_sales = pd.read_csv('/content/drive/My Drive/data/train.csv')
df_sales.shape

df_sales.head(10)

df_sales['date'] = pd.to_datetime(df_sales['date'])
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])

df_sales = df_sales.groupby('date').sales.sum().reset_index()

df_sales.head()

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(df_sales['date'],df_sales['sales'])
# plt.plot(y,sales)
​
plt.show()

df_diff = df_sales.copy()

df_diff['prev_sales'] = df_diff['sales'].shift(1)

​
df_diff.head()

#drop the null values and calculate the difference
df_diff = df_diff.dropna()

df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])

df_diff.head(10)

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot( df_diff['date'],df_diff['diff'])
#plt.figure(figsize=(20000,10000))
​

df_supervised = df_diff.drop(['prev_sales'],axis=1)

for inc inrange(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)

df_supervised.head(10)

df_supervised.tail(6)

df_supervised = df_supervised.dropna().reset_index(drop=True)

import statsmodels.formula.api as smf 
​
model = smf.ols(formula='diff ~ lag_1', data=df_supervised)
​
model_fit = model.fit()
​
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


import statsmodels.formula.api as smf 
​
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5', data=df_supervised)
​
# Fit the regression
model_fit = model.fit()
​
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)
import statsmodels.formula.api as smf 
​
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)
​
# Fit the regression
model_fit = model.fit()
​
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','date'],axis=1)
​

#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

df_model.info()

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)
y_pred
y_test
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

pred_test_set = []
for index inrange(0,len(y_pred)):
print (np.concatenate([y_pred[index],X_test[index]],axis=1))
  pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-7:].date)
act_sales = list(df_sales[-7:].sales)
for index inrange(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
df_result
df_sales.head()

#merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='date',how='left')

df_sales_pred

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(df_sales_pred['date'],df_sales_pred['sales'])
plt.plot(df_sales_pred['date'],df_sales_pred['pred_value'])
#plt.figure(figsize=(20000,10000))