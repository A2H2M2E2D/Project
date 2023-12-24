#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------PRE PROCESSING---------------------------------------------

#Importing the dataset
dataset = pd.read_csv('IRIS.csv')

#Print the head of datasetz
print('data shape:',dataset.shape)
print('**************************************')
print('data head = \n' , dataset.head())
print('------------------------------------------------------------------------')


#Check non values
print("number nan value: \n" , dataset.isna().sum())
print('------------------------------------------------------------------------')

#Handel the nan values
nan = []
for column in dataset.columns:
    if dataset[column].isna().sum():
        nan.append(column)

print("Nan columns:\n" , nan)
print('------------------------------------------------------------------------')

#Fill missing values with mean
for column in nan:
    dataset[column].fillna(dataset[column].mode()[0] , inplace=True)

#Again! check non values
print("Check nan value: \n" , dataset.isna().sum())
print('------------------------------------------------------------------------')

#Encode the string columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['species'] = encoder.fit_transform(dataset['species'])
print('After Encoding: = \n' , dataset.head())
print('------------------------------------------------------------------------')

#Spilt the target from the data
x = dataset.drop('species', axis=1)
y = dataset['species']
print('After Spilting (x): = \n' , x)
print('After Spilting (y): = \n' , y)
print('------------------------------------------------------------------------')

#Feature scaling by StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print('After scaling (x) by StandardScaler: = \n' , x)
print('------------------------------------------------------------------------')

#Feature scaling by MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = sc.fit_transform(x)
print('After scaling (x) by MinMaxScaler: = \n' , x)
print('------------------------------------------------------------------------')

#visual scalining
dataset.hist(figsize=(15,15))

#Spliting dataset into (Training set & Test set)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0, stratify=y, shuffle=True)


#---------------------------------------------CLASSIFICATION---------------------------------------------

#Training the k-NN model on the trainig set

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5 , metric='minkowski' , p=2)
classifier.fit(x_train , y_train)

#Prediction the test set result
y_pred = classifier.predict(x_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)

combined_data = np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
plt.figure(figsize=(10, 8))

# Assuming y_test contains actual values and y_pred contains predicted values
plt.hist(y_test, bins=20, color="b", label='Actual Values')
plt.hist(y_pred, bins=20, color="r", histtype="step", label='predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

#Making the confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
print("confusion Matrix = ",cm)
acs = accuracy_score(y_test , y_pred)
print("accuracy_score = ",acs)
print('------------------------------------------------------------------------')

#Visualizing confusion matrix
import seaborn as sns
ax = sns.heatmap(cm , annot=True , cmap = 'Greens')

# Set labels and title
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Actual Value')
ax.set_title('Confusion Matrix')

#Display
plt.show()

