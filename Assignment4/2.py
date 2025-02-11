import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Load the dataset from a CSV file
data = pd.read_csv('weight-height.csv')

# Select weight as the input (X) and length as the target (Y)
weights = data[['Weight']]
lengths = data[['Height']]

# Calculate the average weight and length
avg_weight = np.mean(weights)
avg_length = np.mean(lengths)

# Create and train a simple linear regression model
model = LinearRegression()
model.fit(weights, lengths)

# Predict lengths based on the trained model
predicted_lengths = model.predict(weights)

# Visualize the data with a scatter plot
plt.scatter(weights, lengths, label='Actual Data', alpha=0.5)
plt.scatter(avg_weight, avg_length, color='red', label='Average Point')  # Highlight the average point
plt.plot(weights, predicted_lengths, color='blue', label='Best Fit Line')  # Show the trend line
plt.title('Relationship Between Weight and Length')
plt.xlabel('Weight')
plt.ylabel('Length')
plt.legend()
plt.show()

# Evaluate model performance with RMSE and R2 Score
rmse = np.sqrt(metrics.mean_squared_error(lengths, predicted_lengths))
r2 = metrics.r2_score(lengths, predicted_lengths)

# Print out evaluation results
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-Squared Score (R2): {r2:.2f}')
