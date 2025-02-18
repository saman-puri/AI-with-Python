import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
data = load_diabetes(as_frame=True)
df = data['frame']

# Visualize correlation between features using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr().round(2), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Select 'bmi' and 's5' as independent variables and 'target' as dependent variable
X = df[['bmi', 's5']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on training data
y_train_pred = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
print(f'RMSE (bmi & s5) - Train: {rmse_train:.2f}')
print(f'R² (bmi & s5) - Train: {r2_train:.2f}')

# Evaluate the model on testing data
y_test_pred = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
print(f'RMSE (bmi & s5) - Test: {rmse_test:.2f}')
print(f'R² (bmi & s5) - Test: {r2_test:.2f}')

# Adding 'bp' to the model
X_extended = df[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=5)

# Retrain the model with additional feature
model.fit(X_train, y_train)

# Evaluate the updated model on training data
y_train_pred = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
print(f'RMSE (bmi, s5 & bp) - Train: {rmse_train:.2f}')
print(f'R² (bmi, s5 & bp) - Train: {r2_train:.2f}')

# Evaluate the updated model on testing data
y_test_pred = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
print(f'RMSE (bmi, s5 & bp) - Test: {rmse_test:.2f}')
print(f'R² (bmi, s5 & bp) - Test: {r2_test:.2f}')

# Conclusion
# Based on the RMSE and R² scores, the model with three features ('bmi', 's5', and 'bp') has a lower RMSE and 
# a higher R² on both the training and testing data. This suggests a slight improvement with the addition of 
# 'bp', though not a significant one. The relatively low R² scores indicate that the model does not perform 
# well overall. Additionally, the high RMSE values suggest a considerable difference between the actual and 
# predicted values. However, since there is no significant difference in RMSE between the training and testing
#  data, the model does not exhibit overfitting.
'''

'''
# From the heatmap, 's4' is the fourth most correlated feature with the target (0.43). If we were to add a 
# fourth feature, it would be 's4'. However, since its correlation is weak (<0.5), adding it may not significantly 
# enhance the model's performance. Additionally, it is crucial to assess the model for potential overfitting.'''
