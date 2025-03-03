import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Load CSV file with delimiter ';'
df = pd.read_csv('bank.csv', delimiter=';')

# Select specific columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Convert categorical variables into dummy/indicator variables
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

#convert target variable y to binary (yes=1, no=0)
df3['y'] = df3['y'].map({'yes':1, 'no':0})

# Create the heatmap of correlation matrix
plt.figure(figsize=[12, 8])
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap')
plt.show()

#define target variable and independent variable
y = df3['y']
X = df3.drop(columns=['y'])

#splite the data base into traning sets
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.25)

#train logistic regression and predict
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
y_pred_log = log_model.predict(X_test)

#print confusion matrix and accuracy

confusion_matrix_log = confusion_matrix(y_test,y_pred_log)
print(f'Confusion matrix:{confusion_matrix_log}')

accuracy_score_log = accuracy_score(y_test, y_pred_log)
print(f'Accuracy Score: {accuracy_score_log}')


#Heat map for confusion matrix
plt.figure(figsize=[5, 4])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues',fmt='d',xticklabels=['no','yes'], yticklabels=['no','yes'])
plt.title('Heatmap')
plt.show()


#train and predict using kneighbours K= 3

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)
y_pred_knn = knn_model.predict(X_test)

#print confusion matrix
confusion_matrix_knn = confusion_matrix(y_test,y_pred_knn)
print(f'Confusion matrix knn:{confusion_matrix_knn}')

accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy Score(knn): {accuracy_score_knn}')

plt.figure(figsize=[5, 4])
sns.heatmap(confusion_matrix_knn, annot=True, cmap='Reds',fmt='d',xticklabels=['no','yes'], yticklabels=['no','yes'])
plt.title('Heatmap of confusion matrix')
plt.show()


# Compare results

print('\nComparision between models:')
print(f'Logistic Regression Accuracy :{accuracy_score_log:4f}')
print(f'KNN Accuracy(K=3):{accuracy_score_knn:4f}')


'''
#Findings:
        -Logistic regression accuracy performs well in data sets than KNN accuracy.
        -The KNN model (k=3) exhibits less accuracy together with potential dimensionality issues in this case.
        -The performance of KNN can be enhanced by trying multiple values of 'k' and implementing feature scaling.
        -The interpretability with shorter runtimes of Logistic Regression surpasses those of KNN.
'''

