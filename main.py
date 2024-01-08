# import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# open files
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print(data_train.head(5))

# preprocessing data
data_train['Age'].fillna(data_train['Age'].median(), inplace=True)
data_test['Age'].fillna(data_test['Age'].median(), inplace=True)
data_train['Embarked'].fillna('S', inplace=True)
data_test['Fare'].fillna(data_test['Fare'].median(), inplace=True)

women = data_train[data_train['Sex'] == 'female']['Survived']
rate_women = sum(women)/len(women)
print(f'rate_women: {rate_women}')

men = data_train[data_train['Sex'] == 'male']['Survived']
rate_men = sum(men)/len(men)
print(f'rate_men: {rate_men}')

# modeling and training
classes = ['Pclass', 'Sex', 'SibSp', 'Parch']

y = data_train['Survived']
X = pd.get_dummies(data_train[classes])

X_test = pd.get_dummies(data_test[classes])

model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=1)
model.fit(X, y)

# make predictions
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)