
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'pass_exam': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['hours_studied', 'attendance']]
y = df['pass_exam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Predict new sample
hours = float(input("Enter hours studied: "))
att = float(input("Enter attendance percentage: "))
result = model.predict([[hours, att]])
print("Result: PASS" if result[0] == 1 else "FAIL")
