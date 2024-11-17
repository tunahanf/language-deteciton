import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = load_dataset("FrancophonIA/language_detection")
df = dataset["train"].to_pandas()
df = df[(df["Language"] == "English") | (df["Language"] == "Turkish") | (df["Language"] == "Spanish") | (df["Language"] == "German") | (df["Language"] == "Italian")]
x = np.array(df["Text"])
y = np.array(df["Language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.42, random_state=52)
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)
userText = input("Enter your phrase : ")
data = cv.transform([userText]).toarray()
output = model.predict(data)
print("The language of the phrase is --", output, "--")