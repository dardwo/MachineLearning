import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

FEATURES = ["Age", "Sex_female", "Sex_male", "Fare", "Pclass", "Parch", "SibSp"]

alldata = pd.read_csv(
    "titanic.tsv",
    header=0,
    sep="\t",
    usecols=[
        "Survived",
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked"
    ],
)

alldata = alldata.drop(['Cabin', 'Ticket'], axis='columns')

alldata['Age'] = alldata['Age'].fillna('{:.2f}'.format(alldata['Age'].mean()))

alldata['Fare'] = alldata['Fare'].apply('{:.2f}'.format)

alldata['Name'] = alldata['Name'].apply(lambda x: re.search(r'\b([A-Za-z]+)\.', x).group(1) if pd.notna(x) else x)

alldata = pd.get_dummies(alldata, columns=['Sex', 'Embarked', 'Name'])

#alldata.to_csv('outputTitanic.tsv', sep='\t', index=False)

data_train, data_test = train_test_split(alldata, test_size=0.2)

y_train = pd.DataFrame(data_train["Survived"])
x_train = pd.DataFrame(data_train[FEATURES])
model = LogisticRegression()
model.fit(x_train, y_train)

y_expected = pd.DataFrame(data_test["Survived"])
x_test = pd.DataFrame(data_test[FEATURES])
y_predicted = model.predict(x_test)

precision, recall, fscore, support = precision_recall_fscore_support(
    y_expected, y_predicted, average="micro"
)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {fscore}")

score = model.score(x_test, y_expected)

print(f"Model score: {score}")