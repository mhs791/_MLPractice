import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

df = pd.read_csv('../Datasets/diabetes.csv')
x = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')
print(f'Train Accuracy: {accuracy_train * 100:.2f}%')

conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
print(f'Test Confusion Matrix:\n{conf_matrix_test}')
print(f'Train Confusion Matrix:\n{conf_matrix_train}')

recall_test = recall_score(y_test, y_pred_test)
recall_train = recall_score(y_train, y_pred_train)
print(f'Test Recall: {recall_test * 100:.2f}%')
print(f'Train Recall: {recall_train * 100:.2f}%')

precision_test = precision_score(y_test, y_pred_test)
precision_train = precision_score(y_train, y_pred_train)
print(f'Test Precision: {precision_test * 100:.2f}%')
print(f'Train Precision: {precision_train * 100:.2f}%')