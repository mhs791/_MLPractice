import numpy as np

salary = np.array([2500, 3200, 5000 , 5500, 5800, 6000, 6500, 7000])
expriance = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=2)
model.fit(expriance, salary)
y_predict = model.predict(expriance)

import matplotlib.pyplot as plt

plt.plot(expriance, salary, marker="*", color = "blue", label="Actual Salary")
plt.plot(expriance, y_predict, color = "red", label="Predicted Salary")
plt.title("Decision Tree Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()