from sklearn.datasets import load_breast_cancer
df = load_breast_cancer()

x = df.data
y = df.target

print(x,y)