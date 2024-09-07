import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.conftest import pyplot
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

# Read in the dataset and define parameters we want the model to look at
data = pd.read_csv('student-mat.csv', sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop(predict, axis=1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""
# Train the model multiple times to find one with high accuracy
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
            f.close()
"""

# Open our saved model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
for i in range(0, len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = "G1"
style.use('ggplot')
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()