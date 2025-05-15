
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('synthetic_webdev_dataset.csv')
X = df[['num_frontend_components']].values
y = df['estimated_completion_time_hrs'].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Print coefficients
print(f"Model Coefficients: w = {model.coef_[0]}, b = {model.intercept_}")

# Predict and plot
y_pred = model.predict(X_scaled)
plt.scatter(X, y, label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel("Number of Frontend Components")
plt.ylabel("Estimated Completion Time (hrs)")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()
