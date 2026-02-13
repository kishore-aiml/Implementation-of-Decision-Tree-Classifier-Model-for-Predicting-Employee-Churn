# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Random Forest
2. Logistic Regression
3. Extreme Gradient Boosting (XGBoost)
4. Support Vector Machine (SVM)

## Program:
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = {
    "Age": [22, 35, 45, 28, 50, 41, 30, 26, 48, 33],
    "Salary": [20000, 50000, 70000, 30000, 90000, 65000, 40000, 25000, 85000, 48000],
    "YearsAtCompany": [1, 7, 15, 2, 20, 10, 4, 1, 18, 6],
    "JobSatisfaction": [2, 4, 3, 2, 5, 4, 3, 1, 5, 3],
    "Churn": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary", "YearsAtCompany", "JobSatisfaction"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Age", "Salary", "YearsAtCompany", "JobSatisfaction"],
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier for Employee Churn Prediction")
plt.show()

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHORE J
RegisterNumber:  212225240072
*/
```

## Output:
<img width="458" height="186" alt="Screenshot 2026-02-13 142035" src="https://github.com/user-attachments/assets/1b47dcd4-e71d-4cfa-ab34-d4e811975c3b" />

<img width="943" height="614" alt="Screenshot 2026-02-13 142101" src="https://github.com/user-attachments/assets/c108efd6-d1bc-4b2f-b0b5-d187e405988b" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
