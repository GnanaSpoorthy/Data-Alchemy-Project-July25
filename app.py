# app.py --- Simple Streamlit Dashboard for laptops_dataset.csv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Laptops Dashboard", layout="wide")
st.title("💻 Laptops Reviews — Simple Dashboard")

# =============== DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("laptops_dataset.csv")   # <-- fixed file name
    # light cleanup for your columns (if present)
    for col in ["overall_rating", "no_ratings", "no_reviews", "rating"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)
    if "review" in df.columns:
        df["review"] = df["review"].fillna("No Review")
        df["review_len"] = df["review"].astype(str).apply(lambda x: len(x.split()))
    return df

df = load_data()

st.write("### Preview of dataset")
st.dataframe(df.head(), use_container_width=True)

num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("No numeric columns found in dataset.")
    st.stop()

# =============== EDA =================
st.subheader("Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)
with col1:
    hist_col = st.selectbox("Histogram column", num_cols, index=0)
    st.plotly_chart(px.histogram(df, x=hist_col, nbins=40, title=f"Histogram: {hist_col}"),
                    use_container_width=True)

with col2:
    if len(num_cols) >= 2:
        x_col = st.selectbox("Scatter X", num_cols, index=0)
        y_col = st.selectbox("Scatter Y", [c for c in num_cols if c != x_col], index=0)
        # removed trendline="ols"
        st.plotly_chart(px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}"),
                        use_container_width=True)

if len(num_cols) >= 2:
    st.plotly_chart(px.imshow(df[num_cols].corr(numeric_only=True),
                              text_auto=True, aspect="auto",
                              title="Correlation Heatmap"),
                    use_container_width=True)

# =============== MODEL =================
st.subheader("Modeling")

# pick target
target_default = num_cols.index("overall_rating") if "overall_rating" in num_cols else 0
target = st.selectbox("Target (numeric)", num_cols, index=target_default)
features = [c for c in num_cols if c != target]
st.caption(f"Using features: {features}")

X = df[features]
y = df[target]

test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model_choice = st.radio("Model", ["LinearRegression", "DecisionTreeRegressor"], horizontal=True)

if model_choice == "LinearRegression":
    model = LinearRegression()
else:
    max_depth = st.selectbox("max_depth (tree)", [None, 3, 5, 10], index=0)
    min_leaf = st.selectbox("min_samples_leaf (tree)", [1, 5, 10], index=0)
    model = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_leaf)

# train
model.fit(X_train, y_train)
pred = model.predict(X_test)

# metrics
r2   = r2_score(y_test, pred)
mae  = mean_absolute_error(y_test, pred)
mse  = mean_squared_error(y_test, pred)
rmse = float(np.sqrt(mse))

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{r2:.4f}")
m2.metric("MAE", f"{mae:.4f}")
m3.metric("MSE", f"{mse:.4f}")
m4.metric("RMSE", f"{rmse:.4f}")

st.plotly_chart(
    px.scatter(x=y_test, y=pred, labels={"x": "Actual", "y": "Predicted"},
               title="Predicted vs Actual"),
    use_container_width=True
)

# interpretation
if model_choice == "LinearRegression":
    st.write("#### Coefficients")
    st.dataframe(pd.DataFrame({"feature": features, "coef": model.coef_})
                 .sort_values("coef", key=np.abs, ascending=False),
                 use_container_width=True)
else:
    st.write("#### Feature Importances")
    st.plotly_chart(
        px.bar(x=features, y=model.feature_importances_,
               labels={"x": "feature", "y": "importance"},
               title="Feature Importances (Tree)"),
        use_container_width=True
    )

st.caption("Run with:  streamlit run app.py")