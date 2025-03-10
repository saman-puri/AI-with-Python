import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#read the csv file 
df = pd.read_csv('data_banknote_authentication.csv')

#select target variables and feature variables as X
y = df['class']
X = df.drop(columns=['class'])

#split the data into traning and test sets
x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=20)

#train SVM with linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(x_train,y_train)

#predict and evaluate linear SVM
y_pred_linear = svm_model.predict(x_test)
print("Confusion Matrix (Linear Kernel):\n", confusion_matrix(y_test, y_pred_linear))
print("Classification Report (Linear Kernel):\n", classification_report(y_test, y_pred_linear))

svm_model_rvf= SVC(kernel='rvf')
svm_model.fit(x_train,y_train)

#predict and evaluate RVF SVM
y_pred_rvf = svm_model.predict(x_test)
print("confusion Matrix (rvf kernel):\n",confusion_matrix(y_test,y_pred_rvf))
print("Classification Report (rvf kernel):\n",classification_report(y_test,y_pred_rvf))


# Comparison
'''  
 -Linear kernel achieves effective results on datasets that can be divided apart using a straight line or any
 hyperplane dimension.
-The RBF kernel demonstrates greater flexibility by expanding data dimensions to detect intricate relationships in the dataset. Non-linear class separability
in datasets tends to produce better performance from the RBF kernel.
-The model that produces higher accuracy ratings plus better precision/recall/F1-score will be the optimal
selection for classifying this particular dataset.'''
