# 📸 Instagram Post Performance Predictor

A serverless ML-powered app to predict likes, comments, and sentiment for Instagram posts — using either manual input or public IG post links.

## 🚀 Features

- Predict Likes & Comments for IG posts
- Sentiment Analysis (TextBlob)
- Visual dashboard with charts
- Token-based access control
- Export predictions to CSV/PDF
- Serverless deployment on Streamlit Cloud

## 🧰 Tech Stack

- Python + Streamlit
- Scikit-learn (RandomForest)
- Instaloader (scraping)
- TextBlob (sentiment)
- Plotly (visuals)

## 📦 Setup Locally

```bash
git clone https://github.com/Aji6646/Insta_ML_Dashboard.git
cd instagram-dashboard
pip install -r requirements.txt
streamlit run dashboard.py
