import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load the data
df = pd.read_csv("poly_regression/fuel_consumption_vs_speed.csv")

X = df[['speed_kmh']].values
y = df['fuel_consumption_l_per_100km'].values

# 2. Try polynomial degrees from 1 to 5
degrees = range(1, 6)
mse_list = []
mae_list = []

plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='black', label='Actual data')

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    mse_list.append(mse)
    mae_list.append(mae)

    # Plot prediction
    X_plot = np.linspace(min(X)[0], max(X)[0], 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f'Degree {degree} (MSE: {mse:.2f})')

plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.title('Polynomial Regression for Fuel Consumption')
plt.legend()
plt.grid(True)
plt.show()

# 3. Show error table
for i, deg in enumerate(degrees):
    print(f"Degree {deg}: MSE = {mse_list[i]:.3f}, MAE = {mae_list[i]:.3f}")

# 4. Choose best degree (lowest MSE)
best_degree = degrees[np.argmin(mse_list)]
print(f"\nBest degree: {best_degree}")

# 5. Final model and prediction for 35, 95, 140 km/h
final_model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
final_model.fit(X, y)

speeds_to_predict = np.array([[35], [95], [140]])
predicted_consumption = final_model.predict(speeds_to_predict)

for i, speed in enumerate(speeds_to_predict.flatten()):
    print(f"Predicted fuel consumption at {speed} km/h: {predicted_consumption[i]:.2f} L/100km")
