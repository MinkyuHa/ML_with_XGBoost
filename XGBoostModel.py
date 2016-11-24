import numpy
import xgboost
#from sklearn import cross_validation  #deprecated
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=",")

#split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]

#split data into train and test sets
seed = 7
test_size = 0.33
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed) #deprecated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

print(model)

#make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#evaluate prediction
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


