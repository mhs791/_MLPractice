import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA

df = pd.read_csv('../Datasets/diabetes.csv')

x = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=25, max_iter=300 , random_state=42)
model.fit(x_train,y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

acc_train = accuracy_score(y_train,y_pred_train)
acc_test = accuracy_score(y_test,y_pred_test)

r_train = recall_score(y_train,y_pred_train)
r_test =  recall_score(y_test,y_pred_test)

p_train = precision_score(y_train,y_pred_train)
p_test = precision_score(y_test,y_pred_test)

conf_train = confusion_matrix(y_train,y_pred_train)
conf_test = confusion_matrix(y_test,y_pred_test)

print(
f'acc_train:{acc_train*100:.2f}%',
f'\nacc_test:{acc_test*100:.2f}%',
f'\np_train:{p_train*100:.2f}%',
f'\np_test:{p_test*100:.2f}%',
f'\nr_train:{r_train*100:.2f}%',
f'\nr_test:{r_test*100:.2f}%',
f'\nconf_train:\n{conf_train}',
f'\nconf_test:\n{conf_test}'
)

pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)

x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

model_pca = MLPClassifier(hidden_layer_sizes=25, max_iter=300 , random_state=42)
model_pca.fit(x_train_pca,y_train)

y_pred_train_pca = model_pca.predict(x_train_pca)
y_pred_test_pca = model_pca.predict(x_test_pca)

acc_train_pca = accuracy_score(y_train,y_pred_train_pca)
acc_test_pca = accuracy_score(y_test,y_pred_test_pca)

r_train_pca = recall_score(y_train,y_pred_train_pca)
r_test_pca =  recall_score(y_test,y_pred_test_pca)

p_train_pca = precision_score(y_train,y_pred_train_pca)
p_test_pca = precision_score(y_test,y_pred_test_pca)

conf_train_pca = confusion_matrix(y_train,y_pred_train_pca)
conf_test_pca = confusion_matrix(y_test,y_pred_test_pca)

print("\nWith PCA:")
print(
f'acc_train:{acc_train_pca*100:.2f}%',
f'\nacc_test:{acc_test_pca*100:.2f}%',
f'\np_train:{p_train_pca*100:.2f}%',
f'\np_test:{p_test_pca*100:.2f}%',
f'\nr_train:{r_train_pca*100:.2f}%',
f'\nr_test:{r_test_pca*100:.2f}%',
f'\nconf_train:\n{conf_train_pca}',
f'\nconf_test:\n{conf_test_pca}'
)
