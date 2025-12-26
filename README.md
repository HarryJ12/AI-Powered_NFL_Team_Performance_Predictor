# NextDown: AI-Powered NFL Team Performance Predictor

NextDown is an NFL analytics pipeline made primarily with Python that uses multi-season team data and machine learning to predict **points scored** and **win probability**, delivered through an interactive dashboard.


## Overview
-  Used Colab to clean and analyze **relevant seasons of NFL team statistics** found on [Kaggle](https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn/data)

        
-   Trained ML models using engineered features for:
    
    -   **Regression:** expected points
        
    -   **Classification:** win/loss probability
        
-   Used a **Vegas-style meta-model** to align score predictions with win probabilities
    
-   Visualized results with **Plotly** found in supporting summary documents 

- Created a **Streamlit** dashboard with extra features and  useability

## Modeling

**Base Models**

-   Linear & Logistic Regression

**Meta-Model (Vegas-Style)**
    
-   Applies a calibrated spread-like adjustment so both models agree
    
-   Mimics how sportsbooks reconcile totals and win odds
    

**Evaluation**

-   Regression: RMSE, R²
    
-   Classification: Accuracy, F1, Brier, ROC-AUC
    
-   Trained on earlier seasons, tested on the most recent season

## Dashboard

-   Team selector
    
-   Predicted points, win probability, and meta-adjusted score
    
-   Historical performance querying

## Tech Stack

Python · Pandas · Scikit-learn · Streamlit · Plotly · Colab

## Run Locally

**Clone repository**
```git clone https://github.com/your-username/NextDown.git
cd NextDown
```

**Create virtual environment**
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

**Install dependencies**
```
pip install -r requirements.txt
```

**Launch dashboard**
```
streamlit run app.py
```