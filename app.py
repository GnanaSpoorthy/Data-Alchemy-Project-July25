# app.py — Laptop Reviews Dashboard (collapsible + results download)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Laptop Reviews Dashboard", layout="wide")
st.title("💻 Laptop Reviews — Interactive Dashboard")

# ---------- Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("laptops_dataset.csv")  # keep CSV in same folder as this file
    # light cleaning for common columns
    for col in ["overall_rating", "no_ratings", "no_reviews", "rating"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)
    if "review" in df.columns:
        df["review"] = df["review"].fillna("No Review")
        df["review_len"] = df["review"].astype(str).apply(lambda x: len(x.split()))
    return df

df = load_data()
num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.error("No numeric columns found in laptops_dataset.csv")
    st.stop()

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")

# try to detect a brand/company column if present
brand_col = next((c for c in df.columns if c.lower() in ["brand", "company", "manufacturer"]), None)
if brand_col:
    brands = sorted(df[brand_col].dropna().astype(str).unique().tolist())
    sel_brands = st.sidebar.multiselect(
        "Brand",
        options=brands,
        default=brands[:10] if len(brands) > 10 else brands
    )
else:
    sel_brands = None

text_query = st.sidebar.text_input("Search in title/review", "")

range_filters = {}
for c in num_cols:
    cmin, cmax = float(np.nanmin(df[c])), float(np.nanmax(df[c]))
    range_filters[c] = st.sidebar.slider(f"{c} range", cmin, cmax, (cmin, cmax))

# apply filters
mask = pd.Series(True, index=df.index)
if brand_col and sel_brands:
    mask &= df[brand_col].astype(str).isin(sel_brands)

if text_query:
    cols_text = [c for c in df.columns if c.lower() in ["title", "review"]]
    if cols_text:
        sub = False
        for c in cols_text:
            sub = sub | df[c].astype(str).str.contains(text_query, case=False, na=False)
        mask &= sub

for c, (lo, hi) in range_filters.items():
    mask &= df[c].between(lo, hi)

df_f = df[mask].copy()

# ---------- Collapsible: Data Preview ----------
with st.expander("📄 Data Preview", expanded=True):
    st.caption(f"Showing {len(df_f):,} of {len(df):,} rows after filters.")
    st.dataframe(df_f.head(20), use_container_width=True)

# ---------- Collapsible: EDA ----------
with st.expander("📊 Exploratory Data Analysis (EDA)", expanded=True):
    left, right = st.columns(2)
    with left:
        hist_col = st.selectbox("Histogram column", num_cols, index=0, key="hist")
        st.plotly_chart(
            px.histogram(df_f, x=hist_col, nbins=40, title=f"Histogram: {hist_col}"),
            use_container_width=True
        )

    with right:
        if len(num_cols) >= 2:
            x_col = st.selectbox("Scatter X", num_cols, index=0, key="sx")
            y_candidates = [c for c in num_cols if c != x_col]
            y_col = st.selectbox("Scatter Y", y_candidates, index=0, key="sy")
            st.plotly_chart(
                px.scatter(df_f, x=x_col, y=y_col, title=f"{x_col} vs {y_col}"),
                use_container_width=True
            )

    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap"),
            use_container_width=True
        )

# ---------- Collapsible: Modeling ----------
with st.expander("🤖 Modeling", expanded=False):
    target_default = num_cols.index("overall_rating") if "overall_rating" in num_cols else 0
    target = st.selectbox("Target (numeric)", num_cols, index=target_default)
    features = [c for c in num_cols if c != target]
    st.caption(f"Using features: {features}")

    if len(df_f) < 5:
        st.warning("Not enough rows after filtering to train a model. Loosen filters.")
    else:
        X = df_f[features]
        y = df_f[target]

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model_choice = st.radio("Model", ["LinearRegression", "DecisionTreeRegressor"], horizontal=True)

        if model_choice == "LinearRegression":
            model = LinearRegression()
        else:
            max_depth = st.selectbox("max_depth (tree)", [None, 3, 5, 10], index=0)
            min_leaf = st.selectbox("min_samples_leaf (tree)", [1, 5, 10], index=0)
            model = DecisionTreeRegressor(
                random_state=42, max_depth=max_depth, min_samples_leaf=min_leaf
            )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # metrics (version-safe)
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
            px.scatter(x=y_test, y=pred, labels={"x": "Actual", "y": "Predicted"}, title="Predicted vs Actual"),
            use_container_width=True
        )

        # === Actual vs Predicted table + download ===
        results = pd.DataFrame({
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(pred).reset_index(drop=True)
        })
        results["Error"] = results["Predicted"] - results["Actual"]
        results["AbsError"] = results["Error"].abs()

        st.write("#### Actual vs Predicted")
        st.dataframe(results.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Actual vs Predicted (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="actual_vs_predicted.csv",
            mime="text/csv"
        )

        # interpretation
        if model_choice == "LinearRegression":
            st.write("#### Coefficients")
            coef_df = pd.DataFrame({"feature": features, "coef": model.coef_}).sort_values(
                "coef", key=np.abs, ascending=False
            )
            st.dataframe(coef_df, use_container_width=True)
        else:
            st.write("#### Feature Importances")
            imp_df = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            st.plotly_chart(
                px.bar(imp_df, x="feature", y="importance", title="Feature Importances"),
                use_container_width=True
            )

# ---------- Collapsible: Download filtered data ----------
with st.expander("⬇️ Download filtered data", expanded=False):
    st.download_button(
        "Download CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="laptops_filtered.csv",
        mime="text/csv"
    )

st.caption("Tip: click a heading to expand/collapse a section • Run with:  streamlit run app.py")