# 🏦 Credit Card Fraud Detection - Exploratory Data Analysis (EDA)

This project focuses on **data cleaning, transformation, and visualization** of a real-world credit card transaction dataset. The goal is to explore patterns, detect anomalies, and prepare the dataset for potential fraud detection modeling (without using ML libraries).

---

## 📌 Project Objectives
- Perform **data cleaning** (remove duplicates, handle missing values).  
- Apply **data transformations** such as scaling, log transformation, and time feature engineering.  
- Detect **outliers** using the IQR method.  
- Explore **class imbalance** between Fraud and Non-Fraud transactions.  
- Visualize patterns using **Matplotlib & Seaborn**.  

---

## 🛠️ Technologies Used
- **Python 3**  
- **Pandas** – data manipulation  
- **NumPy** – numerical operations  
- **Matplotlib & Seaborn** – data visualization  

---

## 📂 Dataset
The dataset used is the **Credit Card Fraud Detection Dataset**.  
It is publicly available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  

⚠️ *The dataset is not uploaded here due to size restrictions. Please download it from Kaggle and place it in your working directory.*  

---

## 🚀 Steps Performed
1. **Data Exploration**
   - Checked shape, summary statistics, and duplicate entries.  
   - Created new features:  
     - `Fraudilent` (Fraud vs Non-Fraud labels)  
     - `Amount-Scaled` (normalized transaction amounts)  
     - `Amount_Log` (log-transformed amounts)  
     - Time features (`Time Hours`, `Time Days`)  

2. **Outlier Detection**
   - Used **IQR method** to detect outliers in transaction amounts.  

3. **Data Visualization**
   - Histogram of transaction amounts.  
   - Bar chart showing Fraud vs Non-Fraud distribution.  
   - Correlation heatmap for `V` features, `Amount`, and `Class`.  

---

## 📊 Key Insights
- Fraud transactions are **highly imbalanced (~0.17%)** compared to non-fraud.  
- Some transactions have extreme `Amount` values (outliers).  
- Correlation heatmap highlights which features (`V` variables) are important for fraud detection.  

---

## 📸 Sample Visuals
- Distribution of transaction `Amount`.  
- Fraud vs Non-Fraud transaction counts.  
- Correlation heatmap of features.  

---

## ▶️ How to Run
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-eda.git
   cd credit-card-fraud-eda

---

## 🌟 Conclusion

This project demonstrates my ability to:

Clean and preprocess raw datasets.

Apply transformations and detect outliers.

Visualize and interpret real-world data insights.

✨ This project is part of my Data Science learning journey and showcases my skills for internship opportunities.