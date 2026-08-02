from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("spam.csv", sep='\t', header=None, names=['label','message'])
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Train model
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']
model = MultinomialNB()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['sms']
    sms_vec = vectorizer.transform([sms])
    prediction = model.predict(sms_vec)[0]
    result = "Spam 🚨" if prediction == 1 else "Not Spam ✅"
    return render_template('index.html', prediction=result, sms=sms)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
if __name__ == "__main__":
    app.run(debug=True)
