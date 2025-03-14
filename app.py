import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("TASK-ML-INTERN.csv")
    df.drop(columns=["hsi_id"], inplace=True)
    return df

@st.cache_data
def preprocess_data(df):
    X = df.iloc[:, :-1].fillna(df.median()) 
    y = df.iloc[:, -1]
    
    
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    y = y.loc[X.index]

    return X, y

df = load_data()
X, y = preprocess_data(df)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train models
def train_models():
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_pca, y_train)

    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train_pca, y_train)

    lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    lgbm.fit(X_train_pca, y_train)

    return rf, gb, lgbm

# Train models dynamically
rf_model, gb_model, lgb_model = train_models()

# Predictions
def evaluate_model(model, name):
    y_pred = model.predict(X_test_pca)
    return {
        "Model": name,
        "R² Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
    }

results = [
    evaluate_model(rf_model, "Random Forest"),
    evaluate_model(gb_model, "Gradient Boosting"),
    evaluate_model(lgb_model, "LightGBM"),
]

results_df = pd.DataFrame(results)

# Build Streamlit UI
st.title("DON Prediction using Regression")
st.write("This app dynamically trains and evaluates models.")

# Show dataset
if st.checkbox("Show Data"):
    st.write(df.head())

# Show model results
st.subheader("Model Performance")
st.table(results_df.sort_values(by="R² Score", ascending=False))

# MLP Model Training (Only runs when selected)
if st.checkbox("Train Neural Network (MLP)"):
    st.write("Training MLP Model... This may take some time.")

    mlp = Sequential([
        Dense(512, activation="relu", input_shape=(X_train_pca.shape[1],)),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1),
    ])
    
    mlp.compile(optimizer="adam", loss="mse", metrics=["mae"])
    mlp.fit(X_train_pca, y_train, validation_data=(X_test_pca, y_test), epochs=50, batch_size=32, verbose=1)
    
    y_pred_mlp = mlp.predict(X_test_pca)
    mlp_results = {
        "Model": "MLP Neural Network",
        "R² Score": r2_score(y_test, y_pred_mlp),
        "MAE": mean_absolute_error(y_test, y_pred_mlp),
        "MSE": mean_squared_error(y_test, y_pred_mlp),
    }
    
    st.table(pd.DataFrame([mlp_results]))

# Visualizations
st.subheader("Feature Correlations")
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

st.subheader("Feature Importance (Random Forest)")
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importances.sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
st.pyplot(plt)

