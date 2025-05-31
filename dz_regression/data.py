import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("dz_regression/assets/energy_usage.csv")

print("Data loaded:")
print(df.head())

X = df[['temperature', 'humidity', 'hour', 'is_weekend']]
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.xlabel("Actual Consumption (kWh)")
plt.ylabel("Predicted Consumption (kWh)")
plt.title("Actual vs Predicted Consumption")
plt.grid(True)
plt.show();


mae = mean_absolute_error(y_test, y_pred)
avg = y_test.mean()
percent_error = (mae / avg) * 100

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Average Consumption: {avg:.2f}")
print(f"Percent Error: {percent_error:.2f}%")
