import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

#read CSV file 
df = pd.read_csv('suv.csv')


X = df[['Age','EstimatedSalary']]
y = df['Purchased']


#train and test the dataset into .20 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20)

# Step 3: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train Decision Tree with entropy criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=20)
dt_entropy.fit(X_train, y_train)

# Predict and evaluate Decision Tree (Entropy)
y_pred_entropy = dt_entropy.predict(X_test)
print("Confusion Matrix (Decision Tree - Entropy):\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report (Decision Tree - Entropy):\n", classification_report(y_test, y_pred_entropy))

# Step 5: Train Decision Tree with gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=20)
dt_gini.fit(X_train, y_train)

# Predict and evaluate Decision Tree (Gini)
y_pred_gini = dt_gini.predict(X_test)
print("Confusion Matrix (Decision Tree - Gini):\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report (Decision Tree - Gini):\n", classification_report(y_test, y_pred_gini))


# Comparison
'''  
 -The interpretable modeling system of Decision Trees performs efficiently when hierarchical patterns exist.
-Decision splits base their selection on purity through the implementation of the entropy criterion.
-The selection of splits through the gini criterion focuses on minimizing incorrect predictions by directly reducing decision boundaries' impurities.
-The model demonstrating higher accuracy and more favorable precision and recall and F1-score measurements should be selected for this dataset.
-Performance of models improves with feature scaling since it makes all features get equal treatment.
'''