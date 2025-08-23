# 💻 Laptop Reviews Dashboard

Interactive dashboard to analyze laptop reviews data using **Streamlit**, **Plotly**, and **scikit-learn**.  
This was built as part  in the Data Alchemy Project.

---

## 🚀 Features
- 📊 Interactive EDA (histograms, scatter plots, correlation heatmap)
- 🤖 Simple ML Models:
  - Linear Regression
  - Decision Tree Regressor
- 📈 Model evaluation with R², MAE, MSE, RMSE
- 🔍 Feature importance and regression coefficients

---

## 📂 Dataset
File: `laptops_dataset.csv`  
Columns include:
- `overall_rating` – target variable  
- `no_ratings`, `no_reviews`, `rating` – numeric features  
- `review_len` – engineered feature  

---

## ⚙️ Installation (Run Locally)

Clone the repo and install requirements:

```bash
git clone https://github.com/GnanaSpoorthy/Data-Alchemy-Project-July25.git
cd Data-Alchemy-Project-July25
pip install -r requirements.txt
streamlit run app.py

🌐 Live Demo

👉 Open Streamlit App 
https://data-alchemy-project-july25-y3cwbndh75nsfzzpnswxmy.streamlit.app/


🛠️ Tech Stack
	•	Streamlit
	•	Plotly Express
	•	scikit-learn
	•	pandas, numpy
