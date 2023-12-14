import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

file = pd.read_csv('communities.data', header=None, sep=',')

file = file.drop(3, axis='columns')
file = file.replace('?', pd.NA)
file = file.dropna(axis='columns')

data_train, data_test = train_test_split(file, test_size=0.2)

y_train = pd.DataFrame(data_train[[127]])
x_train = pd.DataFrame(data_train[file.columns[:-1]])
y_expected = pd.DataFrame(data_test[127])
x_test = pd.DataFrame(data_test[file.columns[:-1]])

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_1 = LinearRegression()
model_1.fit(x_train_scaled, y_train)
y_predicted = model_1.predict(x_test_scaled) 

model_2 = Ridge(alpha=1.0)
model_2.fit(x_train_scaled, y_train)
y_pred = model_2.predict(x_test_scaled)

print(f'Bez regulacji: {mean_squared_error(y_expected, y_predicted)}')
print(f'Z Regulacja: {mean_squared_error(y_expected, y_pred)}')