import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , recall_score , precision_score , confusion_matrix

df = pd.read_csv('diabetes.csv')

x = df.drop('Outcome', axis=1)
y =df['Outcome']

x = np.array(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=8)
model.fit(x_train ,y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
p_train = precision_score(y_train,y_pred_train)
p_test = precision_score(y_test, y_pred_test)
r_train = recall_score(y_train, y_pred_train)
r_test = recall_score(y_test, y_pred_test)
conf_train = confusion_matrix (y_train, y_pred_train)
conf_test = confusion_matrix (y_test, y_pred_test)

print(f'acc_train:{acc_train*100:.2f}%',
      f'\nacc_test:{acc_test*100:.2f}%',
      f'\np_train:{p_train*100:.2f}%',
      f'\np_test:{p_test*100:.2f}%',
      f'\nr_train:{r_train*100:.2f}%',
      f'\nr_test:{r_test*100:.2f}%',
      f'\nconf_train:\n{conf_train}',
      f'\nconf_test:\n{conf_test}')