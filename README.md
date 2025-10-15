# Telco Customer Churn Prediction

This project applies **Machine Learning** techniques to predict customer churn in the **Telco Customer Churn** dataset, available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).  
It demonstrates **MLOps best practices** and **reproducible pipelines** using **DVC (Data Version Control)**.

---

## 🎯 Objective

The goal is to identify customers who are likely to cancel their service, enabling proactive retention strategies.  
The project focuses on building a complete, reproducible **machine learning workflow**, including:

- Data preprocessing  
- Model training  
- Model evaluation  
- Data and model versioning with DVC  

---

## 🧱 Project Structure

churn-mlops/
│
├── data/ # Raw and processed datasets (tracked by DVC)
│
├── models/ # Trained models (.pkl)
│
├── src/ # Source code
│ ├── preprocess.py # Data cleaning and transformation
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│ └── utils.py # Helper functions (optional)
│
├── dvc.yaml # DVC pipeline definition
├── dvc.lock # Pipeline lock file
├── .dvc/ # DVC internal metadata
│
├── requirements.txt # Python dependencies
└── README.md # This file

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **pandas**, **numpy** — data manipulation and preprocessing  
- **scikit-learn** — machine learning modeling and metrics  
- **DVC** — data versioning and pipeline automation  
- **Git** — source code version control

---

## 🔁 Pipeline Overview

The project’s workflow is automated with **DVC** and organized into three main stages:

1. **Preprocessing (`preprocess.py`)**  
   - Loads and cleans the Telco dataset  
   - Handles missing values and encodes categorical features  
   - Saves a processed dataset for model training  

2. **Training (`train.py`)**  
   - Trains a model (e.g., Random Forest) on the processed data  
   - Saves the trained model to `models/random_forest.pkl`

3. **Evaluation (`evaluate.py`)**  
   - Calculates key metrics such as *accuracy*, *precision*, *recall*, and *F1-score*  
   - Optionally generates plots or reports for analysis

---

## 🧩 Local DVC Setup

This project uses **local storage** for DVC, ensuring reproducibility without relying on external cloud services.

To rebuild the entire pipeline automatically:

```bash
dvc repro
```

---

🚀 Next Steps

Add logging and performance tracking
Build a simple prediction interface for new customers

---

📚 Dataset

Source: Telco Customer Churn - Kaggle

---

## 🧠 How to Run Locally

```bash
# 1. **Clone the repository** 
   git clone https://github.com/<your-username>/churn-mlops.git
   cd churn-mlops

#2. **Create and activate a virtual environment**
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Linux/Mac:
    source venv/bin/activate

#3. **Install dependencies**
    pip install -r requirements.txt

#4. **Reproduce the entire ML pipeline with DVC**
    dvc repro

```