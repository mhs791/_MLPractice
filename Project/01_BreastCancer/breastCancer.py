from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bc = load_breast_cancer()
X = bc.data
y = bc.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    results = {
        "acc_train": accuracy_score(y_train, y_pred_train),
        "acc_test": accuracy_score(y_test, y_pred_test),
        "precision_train": precision_score(y_train, y_pred_train),
        "precision_test": precision_score(y_test, y_pred_test),
        "recall_train": recall_score(y_train, y_pred_train),
        "recall_test": recall_score(y_test, y_pred_test),
    }
    return results

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

models = {
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=300, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(25,), max_iter=300, random_state=42),
}

results_list = []
for name, model in models.items():
    res = evaluate_model(model, x_train, y_train, x_test, y_test)
    res["Model"] = name
    results_list.append(res)

df_results = pd.DataFrame(results_list)
df_results.set_index("Model", inplace=True)
print(df_results)

df_acc = df_results[["acc_train", "acc_test"]]
df_acc.plot(kind="bar", figsize=(10,6))
plt.title("Accuracy Comparison (Train vs Test)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0,1.05)
plt.legend(["Train", "Test"])
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df_results, annot=True, fmt=".2f", cmap="Blues")
plt.title("Performance Metrics Heatmap")
plt.show()
