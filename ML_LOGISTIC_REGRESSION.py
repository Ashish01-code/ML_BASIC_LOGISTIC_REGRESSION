#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\ashis\Downloads\student_pass_logistic.csv")
print(data.head())


X = data[["StudyHours"]]
y = data["Pass"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


hours = np.array([[4.5]])

prediction = model.predict(hours)
probability = model.predict_proba(hours)

print("Prediction:", prediction[0])
print("Fail probability:", probability[0][0])
print("Pass probability:", probability[0][1])


plt.scatter(X, y)

x_values = np.linspace(X.values.min(), X.values.max(), 100).reshape(-1,1)
probs = model.predict_proba(x_values)[:,1]

plt.plot(x_values, probs, color="red")

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Curve")
plt.show()


# In[ ]:




