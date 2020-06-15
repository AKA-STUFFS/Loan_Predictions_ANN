#Loan_Predictions.ipynb

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

loan=pd.read_csv('loan_final.csv')

X=loan.iloc[:,1:-1].values
y=loan.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [10])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

sc = StandardScaler()
X = sc.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=10,activation='relu'))

ann.add(tf.keras.layers.Dense(units=10,activation='relu'))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train,y_train,batch_size=32,epochs=100)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_pred,y_test)
print(cm)
accuracy_score(y_pred,y_test)
