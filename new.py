import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import io
import os
# Model training
def train_model():
    csv_path = 'heart.csv'
    if not os.path.exists(csv_path):
        csv_path = 'heartproject/heart.csv'
    
    st.write(f"Using file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    st.write("CSV file loaded successfully")
    
    X = df.drop(['target'], axis=1)
    y = df["target"]

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    model.fit(x_train, y_train)
    st.write("Model trained")

    acc = model.score(x_val, y_val)
    st.success(f"Model Accuracy: {acc * 100:.2f}% ")

    return model

st.session_state.model = train_model()