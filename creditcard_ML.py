import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("creditcard.csv")

# czyszczenie danych

duplicates_to_remove = sum(dataset.duplicated()) #pokazuje ile jest duplikatów
print(duplicates_to_remove)
dataset.drop_duplicates(inplace=True)
duplicates_to_remove = sum(dataset.duplicated())
print(duplicates_to_remove)

#sprawdzamy czy są nulle


dataset.drop("Time", axis=1, inplace=True) #pozbywa się kolumny time

# podział data set na części
#1 wartości niezależne x1,x1 ..
X = dataset.iloc[:, dataset.columns != 'Class']

#2 wartosci kolumny decyzyjnej y
y = dataset.Class

#podział danych na treningowe i testowe
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =0.20, random_state=42)

#skalowanie danych treningowych
scl = StandardScaler().fit(x_train)
x_train_scaled = scl.transform(x_train)
x_test_scaled = scl.transform(x_test)

#uczenie modelu
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train_scaled,y_train) # tutaj zopstaje użyty model regresji logistycznej, który wyucza model

#ocena modelu dla danych treningowych
train_accuracy = logistic_regression_model.score(x_train_scaled, y_train)
print(train_accuracy)

#ocena modelu dla danych testowych
y_pred = logistic_regression_model.predict(x_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(test_accuracy)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)