# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 

## AIM:
To Implementat an Auto Regressive Model using Python

## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

## PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/web_traffic.csv', parse_dates=['Timestamp'], dayfirst=True)

df.set_index('Timestamp', inplace=True)
df.sort_index(inplace=True)

data_hourly = df['TrafficCount'].resample('h').sum()
data_hourly = data_hourly.asfreq('h')

result = adfuller(data_hourly)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.8 * len(data_hourly))
train_data = data_hourly.iloc[:x]
test_data = data_hourly.iloc[x:]

lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data_hourly, lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data_hourly, lags=40, alpha=0.05, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)

mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data, label='Actual Traffic Count (Test Set)')
plt.plot(predictions, label='Predicted Traffic Count', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Vehicle Count')
plt.title('AutoReg Model: Predictions vs Actual Web Traffic')
plt.legend()
plt.grid()
plt.show()
```

## OUTPUT:

### Dataset:
![Screenshot (110)](https://github.com/user-attachments/assets/8e0ea3aa-313e-4fbb-87b1-41c8db82915b)

### ADF Test Result:
![Screenshot (109)](https://github.com/user-attachments/assets/78533119-43ea-4c97-9e89-a6eaa5f48eed)

### ACF plot:
![image](https://github.com/user-attachments/assets/549670be-dd8d-42ef-a03c-570afa960c7b)

### PACF plot:
![image](https://github.com/user-attachments/assets/be25ffb6-1efd-4c76-8c5e-852471a7a663)

### Accuracy:
![Screenshot (111)](https://github.com/user-attachments/assets/ee7c0893-f6a5-4b8f-9915-2377a4ca1eb3)

### PERFECTION VS TEST DATA:
![image](https://github.com/user-attachments/assets/81cf3955-8ed3-4dac-803f-cb933c15ee27)

## RESULT:
Thus we have successfully implemented the auto regression function using python.
