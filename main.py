import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

data=pd.read_csv('combined.csv',encoding='latin-1')
data.columns=['label','message']

data_ham=data[data['label']=='ham']
data_spam=data[data['label']=='spam']

data_spam_unsampled=resample(data_spam,replace=True,n_samples=len(data_ham),random_state=42)

data=pd.concat([data_ham,data_spam_unsampled])
data=data.sample(frac=1,random_state=42).reset_index(drop=True)

data['label']=data['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred=model.predict(X_test_vec)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

def predict_sms(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    probability = model.predict_proba(text_vec)

    confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]

    label = "Likely a SPAM 🚫" if prediction[0] == 1 else "Genuine ✅"
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')