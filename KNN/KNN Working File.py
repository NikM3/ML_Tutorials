import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

data = pd.read_csv('car.data')
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

prediction = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(prediction)):
    print("Predicted: ", names[prediction[i]], "Data: ", x_test[i], " Actual: ", names[y_test[i]])
    print("N: ", model.kneighbors([x_test[i]], 9, True))
