import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns

# Load the dataset
data = pd.read_csv('Auto.csv')

# Selecting relevant features for regression analysis
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']
X = data[features]
y = data['mpg']

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Define a range of alpha values for regularization strength
alphas = np.linspace(0, 100, 500)

# Lists to store R² scores for Ridge and Lasso models
r2_ridge = []
r2_lasso = []

# Iterate through different alpha values and compute R² scores
for alpha in alphas:
    # Train and evaluate Ridge Regression
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    r2_ridge.append(r2_score(y_test, ridge_model.predict(X_test)))

    # Train and evaluate Lasso Regression
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    r2_lasso.append(r2_score(y_test, lasso_model.predict(X_test)))

# Plot R² scores for Ridge and Lasso Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alphas, r2_ridge, label='Ridge')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('R² Scores for Ridge Regression')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(alphas, r2_lasso, label='Lasso', color='red')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('R² Scores for Lasso Regression')
plt.legend()

plt.tight_layout()
plt.show()

# Find the best alpha values for Ridge and Lasso
best_alpha_ridge = alphas[np.argmax(r2_ridge)]
best_r2_ridge = max(r2_ridge)
print(f"Best R² for Ridge: {best_r2_ridge:.4f} at Alpha: {best_alpha_ridge:.4f}")

best_alpha_lasso = alphas[np.argmax(r2_lasso)]
best_r2_lasso = max(r2_lasso)
print(f"Best R² for Lasso: {best_r2_lasso:.4f} at Alpha: {best_alpha_lasso:.4f}")

#The best alpha value for Ridge regression is **91.98**, resulting in an **R² score of 0.7717**. For Lasso 
# regression, the optimal alpha is much smaller at **0.2004**, with a **slightly better R² score of 0.7748**.  
# Both models perform similarly, explaining about **77% of the variance** in MPG, which indicates a reasonably
#  good fit. However, Lasso performs **marginally better** than Ridge. The significant difference in optimal
#  alpha values suggests that Ridge requires much stronger regularization (**91.98**) compared to
#  Lasso (**0.2004**) to achieve its best performance.