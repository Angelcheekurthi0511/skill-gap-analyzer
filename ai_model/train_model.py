# ai_model/train_model.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os

# Load job role data
df = pd.read_csv("../data/jobs.csv")

# Clean and format data
df["skills"] = df["skills"].apply(lambda x: [skill.strip().lower() for skill in x.split(',')])
df["role"] = df["role"].str.lower()

# Prepare input and output
df["skill_string"] = df["skills"].apply(lambda x: ' '.join(x))  # Join skills into string for vectorizer
X = df["skill_string"]
y = MultiLabelBinarizer().fit_transform([[role] for role in df["role"]])

# Train model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

pipeline.fit(X, y)

# Save model and encoder
os.makedirs("../models", exist_ok=True)
joblib.dump(pipeline, "../models/job_predictor.pkl")

# Save label encoder separately
role_encoder = MultiLabelBinarizer()
role_encoder.fit([[role] for role in df["role"]])
joblib.dump(role_encoder, "../models/role_encoder.pkl")

print("✅ Model trained and saved in /models/")
