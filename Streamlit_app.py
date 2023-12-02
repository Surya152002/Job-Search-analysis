import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset loading function (replace with actual dataset)
def load_dataset():
    # Placeholder: Load your dataset here
    # For example, a CSV file with columns 'Resume_text' and 'Job_Match'
    data = pd.read_csv('path_to_dataset.csv')
    return data

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Load and preprocess dataset
dataset = load_dataset()
dataset['Processed_Resume'] = dataset['Resume_text'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(dataset['Processed_Resume']).toarray()
y = dataset['Job_Match']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = SGDClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Streamlit app
def main():
    st.title("Job Recommendation System")

    # Resume text input
    resume_text = st.text_area("Enter Your Resume Text")
    if st.button("Recommend Jobs"):
        processed_text = preprocess_text(resume_text)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_text)
        st.write(f"Recommended Job Match: {prediction[0]}")

if __name__ == "__main__":
    main()
