import pandas as pd
import re

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

alldata.to_csv('outputTitanic.tsv', sep='\t', index=False)