import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("spam.csv", sep='\t', header=None, names=['label','message'])

# 2. Encode labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# 4. Convert text to features
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# 7. Test with custom SMS
sample_sms = [
    "Congratulations! You won a free recharge",
    "Hi Raghib, let's meet tomorrow"
]
sample_vec = vectorizer.transform(sample_sms)
preds = model.predict(sample_vec)
print("Predictions (0=ham, 1=spam):", preds)