import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_crop_yield_data():
    df = pd.read_csv("data/yield.csv")
    return df

def preprocess_data(df):
    # Keep only useful columns
    df = df[['Area', 'Item', 'Year', 'Value']].copy()
    df.columns = ['Country', 'Crop', 'Year', 'Yield']

    # Drop missing values
    df.dropna(inplace=True)

    # Remove impossible values
    df = df[df['Yield'] > 0]

    # Encode categorical columns
    le_country = LabelEncoder()
    le_crop = LabelEncoder()
    df['Country_enc'] = le_country.fit_transform(df['Country'])
    df['Crop_enc'] = le_crop.fit_transform(df['Crop'])

    # Features and target
    X = df[['Country_enc', 'Crop_enc', 'Year']].values
    y = df['Yield'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df

def get_classification_data(df):
    """For classification algorithms (KNN, Decision Tree, Random Forest, Logistic Regression)"""
    df = df.copy()

    # Create binary class: high yield vs low yield
    median_yield = df['Yield'].median()
    df['Yield_Class'] = (df['Yield'] > median_yield).astype(int)

    le_country = LabelEncoder()
    le_crop = LabelEncoder()
    df['Country_enc'] = le_country.fit_transform(df['Country'])
    df['Crop_enc'] = le_crop.fit_transform(df['Crop'])

    X = df[['Country_enc', 'Crop_enc', 'Year']].values
    y = df['Yield_Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test