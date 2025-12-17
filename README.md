🎓 SUDHIR TUTORIALS – EduPred AI

EduPred AI is a machine learning–driven educational intelligence platform that predicts student performance and academic risk factors using real-world coaching institute data. The project demonstrates the application of supervised and unsupervised learning models in a production-ready Flask web application.

---

## 📌 Problem Statement

Educational institutions often struggle to identify students at academic risk early, understand performance trends, and take timely interventions. Traditional analysis methods lack predictive intelligence and scalability.

EduPred AI addresses this challenge by leveraging historical student data and machine learning to generate accurate, data-driven academic insights.

---

## 🚀 Solution Overview

EduPred AI trains multiple machine learning models on student data to:
- Predict final academic scores
- Classify student performance levels
- Identify pass/fail outcomes
- Detect dropout and fee-default risks
- Discover learning patterns using clustering

All predictions are strictly model-based and trained on the dataset. No hardcoded decision rules are used during inference.

---

## 🧠 Machine Learning Approach

| Task | Model |
|----|----|
| Final Marks Prediction | Linear Regression |
| Performance Classification | Decision Tree |
| Pass / Fail Prediction | Logistic Regression |
| Dropout Risk Prediction | Decision Tree |
| Fee Default Risk Prediction | Decision Tree |
| Learning Pattern Discovery | K-Means Clustering |

Models are evaluated using accuracy and R² metrics, which are displayed directly in the application dashboard.

---

## 📊 Features & Visual Analytics

- Real-time ML predictions via Flask
- Interactive dashboards using Chart.js
- Student vs institute average comparison
- Skill profiling using radar charts
- Modern UI with Light/Dark mode toggle

---

## 🛠️ Tech Stack

- **Languages:** Python, JavaScript  
- **Backend:** Flask  
- **Machine Learning:** Scikit-Learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Chart.js  
- **Deployment:** GitHub + Render  

---

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py
