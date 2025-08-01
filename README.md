# Accident Risk Spatio-Temporal Analysis and Severity Estimation

This project analyzes over 7 million U.S. traffic accident records (2016–2023) to uncover patterns in severity, identify accident-prone zones, and model predictive insights using clustering and machine learning.

## 🔧 Tools & Technologies
- Python (pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn)
- Clustering: DBSCAN, K-Means
- Classification: Logistic Regression, Random Forest, XGBoost
- Dashboarding: Streamlit, Tableau
- Dataset: [U.S. Accidents Dataset (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

## 📁 Structure
```
accident-risk-analysis/
├── data/                        # Raw/sample datasets (omit large files from repo)
│   └── us_accidents_sample.csv
├── notebooks/                   # Jupyter notebooks for each stage
│   ├── eda_preprocessing.ipynb    # Cleaning & EDA
│   ├── clustering_modeling.ipynb  # DBSCAN & KMeans
│   └── severity_classification.ipynb # ML classification (XGBoost, RandomForest)
├── dashboards/
│   └── streamlit_app.py         # (Optional) Interactive visual dashboard
├── visuals/                     # Charts, maps, and saved visual assets
│   ├── heatmap.png
│   └── confusion_matrix.png
├── requirements.txt             # Python dependencies
├── README.md
└── LICENSE
```

## 📊 Key Features
- Preprocessed, large-scale accident dataset (46+ fields: time, weather, geolocation, etc.)

- Engineered features: hourly, daily, seasonal breakdowns; weather conditions; location specifics

- Predictive severity models: Logistic Regression, Random Forest, XGBoost

- Clustered high-risk zones: Spatial clustering to visualize geographic and temporal hotspots

- Visualization suite: Heatmaps, confusion matrices, accident frequency charts, interactive dashboards

- Actionable insights: Findings for city planners, safety officials, and drivers

## 📄 Final Report
📎 [Capstone Paper (PDF)](https://your-shared-link-here)

## 👤 Lead Contributor
**Isagani Julian Hernandez III**  
University of North Texas, M.S. in Data Science  
Role: Data Collection, Cleaning, Preprocessing, Feature Engineering

---

## 📜 License
MIT License

---
Large/raw datasets are not included due to size—see Kaggle source.

For details on preprocessing, feature engineering, modeling, and results, see the final report above and Jupyter notebooks in /notebooks.

Visualizations and dashboards provide practical insights for improving U.S. road safety based on solid evidence and machine learning
