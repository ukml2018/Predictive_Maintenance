import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from scipy.special import expit

#-- Setting dimension for plot
sns.set(rc={'figure.figsize': (11.7,8.27)})
#-- Load the data
data_pm = pd.read_csv('predictive_maintenance.csv')

#-- Copy the data
data = data_pm.copy()
#print(data_pm.dtypes)

#-- Structure of the Dataset
print(data.info())

#-- Check any NULL col present or not
print(data.isnull().sum())

#-- Summarizing Data
pd.set_option('display.float', lambda x: '%.3f' %x)
pd.set_option('display.max.columns',500)
summary_num = data.describe()
#print(summary_num)
#print(np.unique(data_pm['device']))
#print(data_pm['device'].value_counts())
#-- Summary of categorical variable
#summary_cate = data_pm.describe(include = "0")
#print(summary_cate)

#-- summary of numerical variables
summary_num = data.describe()
print(summary_num)

#-- Frequency of each catagorical variables
#print(data_pm['device'].value_count())

#-- Relation between independent variables
correlation = data_pm.corr()
print("Data Correlation: ", correlation)

#-- Converting date format
def d_convert(dt):
    datetimeobject = datetime.strptime(dt, '%Y%m%d')
    #new_dt = datetimeobject.strftime('%m-%d-%Y')
    return datetimeobject
#data_pm['date'] = d_convert(data_pm['date'])
#print(data_pm['date'])

#-- Do scatter  and count plot
#plt.scatter(data_pm['device'], data_pm['failure'])
#sns.countplot(data_pm['device'],data = data_pm)
#plt.show()

#-- remove insignificant date field
cols = ['date']
new_data = data.drop(cols,axis =1)
new_data = pd.get_dummies(new_data,drop_first=True)
column_list = list(new_data.columns)
print(column_list)

#-- Separating the input name from data
features=list(set(column_list)-set(['failure']))
print(features)

#-- storing the output values in Y
y = new_data['failure'].values
print(y)

#-- storing the input values in x
x = new_data[features].values
print(x)

#-- spliting the data into test and train
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#################################################################################
# Logistic Regression
#################################################################################

#-- Make an instance of the Model
logistic = LogisticRegression()

#-- fitting value for x and y
logistic.fit(train_x,train_y)
pd.set_option('display.max.columns',500)
print("Logistic co-efficient:", logistic.coef_ )
print("Logistic Intercept : ", logistic.intercept_)

#-- predict from the test data
prediction = logistic.predict(test_x)
pd.set_option('display.max.columns',500)
print("LR Prediction: ", prediction)

#-- Calculating the accuracy
accuracy_score = accuracy_score(test_y,prediction)
print('Accuracy Score:', accuracy_score)

#-- Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

# -- Confusion Matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print('Confusion matrix:', confusion_matrix)

#-- Finding the mean of test data value
base_pred = np.mean(test_y)

#-- Repeat the same value till length of test data
base_pred = np.repeat(base_pred, len(test_y))

#-- Finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(test_y,base_pred))
print("RMSE : ", base_root_mean_square_error)

# Fit the classifier
clf = LogisticRegression(C=1e5)
#clf.fit(X, y)
clf.fit(train_x,train_y)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
#plt.scatter(train_x.ravel(), train_y, color='black', zorder=20)
#plt.scatter(train_x, train_y, color='black', zorder=20)
#X_test = np.linspace(-5, 10, 300)

loss = expit(test_x * clf.coef_ + clf.intercept_)
plt.plot(test_x, loss, color='black', linewidth=3)
plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model'),
           loc="lower right", fontsize='small')
plt.tight_layout()
plt.show()
###############################################################
# KNN
###############################################################

# -- Storing the K nearest neighbors classifier
# KNN_classifier = KNeighborsClassifier(n_neighbors=5)

# -- Fitting the value for X and Y
# KNN_classifier.fit(train_x, train_y)

# -- Predict the test value with model
#prediction = KNN_classifier.predict(test_x)
#print("KNN Prediction: ", prediction )

#-- Performance metric check
#confusion_matrix = confusion_matrix(test_y, prediction)
#print("\t","Predicted values")
#print("Original Values ","\n", confusion_matrix)

#-- Calculate the accuracy
#accuracy_score = accuracy_score(test_y, prediction)
#print("KNN Accuracy score: ", accuracy_score)

#print(" KNN Misclassified samples: %d" % (test_y != prediction).sum())

######################################################
# Effect of K value on classifier
######################################################
