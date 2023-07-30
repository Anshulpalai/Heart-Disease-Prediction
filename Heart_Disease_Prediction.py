# Importing all the required libraries

# Loading Dataset
import pandas as pd
# import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# EDA
from collections import Counter

# data preprocessing
from sklearn.preprocessing import StandardScaler

# data splitting
from sklearn.model_selection import train_test_split

# data modelling
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensembling or Combining all
from mlxtend.classifier import StackingCVClassifier

data = pd.read_csv("D:\\Disease_Prediction\\Heart_Disease_Prediction\\Heart_Dataset\\heart.csv")
data.head()
print(data)
data.info()
print(data.describe())
print(data['target'].value_counts())

y = data["target"]
x = data.drop(columns='target', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(y_test.unique())
print(Counter(y_train))

# Model-1 Using Naive Bayes
model1 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(x_train, y_train)
nbpred = nb.predict(x_test)
nb_conf_mat = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("Confusion Matrix of Naive Bayes")
print(nb_conf_mat)
print("\n")
print("Accuracy Score of Naive Bayes Model is: ", nb_acc_score * 100, "\n")  # Gives the accuracy of 85.36
print(classification_report(y_test, nbpred))

# Model-2 Using Logistic Regression
model2 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
lr_conf_mat = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("Confusion Matrix of Logistic Regression")
print(lr_conf_mat)
print("\n")
print("Accuracy score of Logistic Regression Model is: ", lr_acc_score * 100)
print(classification_report(y_test, lr_predict))

# Model-3 Using SVM
model3 = 'Support Vector Machine'
supp_vec = SVC(kernel='rbf', C=2, probability=True)
supp_vec.fit(x_train, y_train)
supp_vec_pred = supp_vec.predict(x_test)
supp_vec_conf_mat = confusion_matrix(y_test, supp_vec_pred)
supp_vec_acc_score = accuracy_score(y_test, supp_vec_pred)
print("Confusion Matrix of SVM")
print(supp_vec_conf_mat)
print("\n")
print("Accuracy Score of SVM Model is: ", supp_vec_acc_score * 100)
print(classification_report(y_test, supp_vec_pred))

# Model-4 Using Random Forest
model4 = 'Random Forest Classifier'
rf = RandomForestClassifier(n_estimators=10, random_state=2, max_depth=1)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_conf_mat = confusion_matrix(y_test, rf_pred)
rf_acc_score = accuracy_score(y_test, rf_pred)
print("Confusion Matrix of Random Forest")
print(rf_conf_mat)
print("\n")
print("Accuracy score of Random Forest Model is: ", rf_acc_score * 100)
print(classification_report(y_test, rf_pred))

# Model-5 K-Neighbors Classification
model5 = 'K-NeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_conf_mat = confusion_matrix(y_test, knn_pred)
knn_acc_score = accuracy_score(y_test, knn_pred)
print("Confusion Matrix of K-NeighborsClassifier")
print(knn_conf_mat)
print("\n")
print("Accuracy score of K-NeighborsClassifier is: ", knn_acc_score * 100)
print(classification_report(y_test, knn_pred))

# Model-6 Using Decision Tree Classifier
model6 = 'Decision Tree Classifier'
dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=1)
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)
dt_conf_mat = confusion_matrix(y_test, dt_pred)
dt_acc_score = accuracy_score(y_test, dt_pred)
print("Confusion Matrix of Decision Tree Classifier")
print(dt_conf_mat)
print("\n")
print("Accuracy score of Decision Tree Classifier is: ", dt_acc_score * 100)
print(classification_report(y_test, dt_pred))

# Model-7 Using Extreme Gradient Boost
model7 = 'Extreme Gradient Boost'
xgb = XGBClassifier(max_dept=15, learning_rate=0.01, n_estimators=25, gamma=0.6, subsample=0.52, colsample_bytree=0.6,
                    seed=27,
                    reg_lambda=2, booster='dart', colsample_bylevel=0.5, colsample_bynode=0.5)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_pred)
xgb_acc_score = accuracy_score(y_test, xgb_pred)
print("Confusion Matrix of XGBClassifier")
print(xgb_conf_matrix)
print("\n")
print("Accuracy score of XGBClassifier is: ", xgb_acc_score * 100)
print(classification_report(y_test, xgb_pred))

imp_feature = pd.DataFrame({'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                         'exang', 'oldpeak', 'slope', 'ca', 'thal' ],
                            'Importance': xgb.feature_importances_})
plt.figure(figsize=(10, 4))
plt.title("Barplot represent feature importance ")
plt.xlabel("Importance")
plt.ylabel("features")
plt.barh(imp_feature['Feature'], imp_feature['Importance'], color=['r', 'g', 'b', 'k'])
# plt.show()

# Using ROC Curve to identify which model performs well
nb_false_positive_rate, nb_true_positive_rate, nb_threshold = roc_curve(y_test, nbpred)
lr_false_positive_rate, lr_true_positive_rate, lr_threshold = roc_curve(y_test, lr_predict)
supp_vec_false_positive_rate, supp_vec_true_positive_rate, supp_vec_threshold = roc_curve(y_test, supp_vec_pred)
rf_false_positive_rate, rf_true_positive_rate, rf_threshold = roc_curve(y_test, rf_pred)
knn_false_positive_rate, knn_true_positive_rate, knn_threshold = roc_curve(y_test, knn_pred)
dt_false_positive_rate, dt_true_positive_rate, dt_threshold = roc_curve(y_test, dt_pred)
xgb_false_positive_rate, xgb_true_positive_rate, xgb_threshold = roc_curve(y_test, xgb_pred)

# Plotting of Roc Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10, 5))
plt.title("Receiver Operating Characterstics Curve")
plt.plot(nb_false_positive_rate, nb_true_positive_rate, label="Naive Bayes Classifier")
plt.plot(lr_false_positive_rate, lr_true_positive_rate, label="Logistic Regression")
plt.plot(supp_vec_false_positive_rate, supp_vec_true_positive_rate, label="Support Vector Classifier")
plt.plot(rf_false_positive_rate, rf_true_positive_rate, label="Random Forest Classifier")
plt.plot(knn_false_positive_rate, knn_true_positive_rate, label='K-Neighbors Classifier')
plt.plot(dt_false_positive_rate, dt_true_positive_rate, label='Decision Tree Classifier')
plt.plot(xgb_false_positive_rate, xgb_true_positive_rate, label="XGBClassifier")
plt.plot([ 0, 1 ], ls='--')
plt.plot([ 0, 0 ], [ 1, 0 ], c='.5')
plt.plot([ 1, 1 ], c='.5')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

# Summarization of all the models performance score
model_ev = pd.DataFrame({'Model': [ 'Logistic Regression', 'Support Vector Classifier', 'Extreme Gradient Boost',
                                    'K-Neighbors Classifier'], 'Accuracy': [lr_acc_score * 100,
                                                                              supp_vec_acc_score * 100,
                                                                            xgb_acc_score * 100,
                                                                              knn_acc_score * 100 ]})
print(model_ev)

colors = [ 'red', 'orange', 'yellow', 'green']
plt.figure(figsize=(12, 5))
plt.title("BarChart Accuracy of Different ML Models")
plt.xlabel("ALgorithms", labelpad=2)
plt.ylabel("% Accuracy")
plt.bar(model_ev[ 'Model' ], model_ev['Accuracy'], color=colors)
plt.show()


# Ensembled Model
scv = StackingCVClassifier(classifiers=[supp_vec, knn, xgb], meta_classifier=supp_vec, random_state=42, use_probas=True)
scv.fit(x_train, y_train)
scv_pred = scv.predict(x_test)
scv_conf_mat = confusion_matrix(y_test, scv_pred)
scv_acc_score = accuracy_score(y_test, scv_pred)
print("Confusion Matrix of Ensembled Model")
print(scv_conf_mat)
print("\n")
print("Accuracy Score of StackingCVClassifier: ", scv_acc_score * 100)
print(classification_report(y_test, scv_pred))

# Accuracy table of different models
model_ev2 = pd.DataFrame({'Model': [ 'LogisticRegression', 'Support Vector Classifier', 'Extreme Gradient Boost',
                                    'K-Neighbors Classifier', 'StackingCVClassifier', 'Random Forest Classifier',
                                     'Naive Bayes Classifier', 'Decision Tree'],
                          'Accuracy': [lr_acc_score * 100, supp_vec_acc_score * 100, xgb_acc_score * 100,
                                        knn_acc_score * 100, scv_acc_score * 100, rf_acc_score*100, nb_acc_score*100,
                                       dt_acc_score*100]})
print(model_ev2)


# # Taking Custom Input
# input_data = (40,1,0,110,167,0,0,114,1,2,1,0,3)
#
# # Now changing the input_data to numpy array
# input_data_as_numpy_array = np.array(input_data)
#
# # Reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#
# prediction = scv.predict(input_data_reshaped)
# print(prediction)
#
# if prediction[0] == 0:
#     print("The Person does not have Heart Disease")
# else:
#     print("The Person has Heart Disease")
#
# # Saving the Trained Model
#
# import pickle
# pickle.dump(scv, open('Ensembled_Model.pkl', 'wb'))
#
# for columns in x:
#     print(columns)

from sklearn import metrics
import matplotlib.pyplot as plt
cm_display = metrics.ConfusionMatrixDisplay(scv_conf_mat, display_labels=[False, True])
cm_display.plot()
plt.show()