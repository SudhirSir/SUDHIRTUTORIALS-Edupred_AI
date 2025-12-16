from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score

# =====================================================
# DATA LOAD
# =====================================================
df = pd.read_excel("FINAL_Coaching_Dataset_CLEAN_v4.xlsx")

df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace("%", "percent", regex=False)
    .str.replace(" ", "_")
)

def find_col(keys):
    for c in df.columns:
        for k in keys:
            if k in c:
                return c
    raise Exception(f"Missing column: {keys}")

science = find_col(["science"])
maths = find_col(["math"])
attendance = find_col(["attendance"])
homework = find_col(["homework"])
attention = find_col(["attention"])
previous = find_col(["previous"])

# =====================================================
# DATA CLEANING
# =====================================================
for c in df.select_dtypes(include=["int64", "float64"]):
    df[c].fillna(df[c].median(), inplace=True)

df[attendance] = df[attendance].astype(str).str.replace("percent", "").astype(float)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["average_marks"] = (df[science] + df[maths]) / 2
df["engagement_score"] = (
    0.4 * df[attendance] +
    0.3 * df[homework] +
    0.3 * df[attention]
)
df["performance_index"] = (
    0.5 * df["average_marks"] +
    0.3 * df["engagement_score"] +
    0.2 * df[previous]
)

df["pass_fail"] = (df["average_marks"] >= 40).astype(int)
df["at_risk"] = (
    (df[attendance] < 70) |
    (df[homework] < 50) |
    (df["average_marks"] < 40)
).astype(int)

# =====================================================
# MODEL TRAINING (6 MODELS)
# =====================================================
features = [science, maths, attendance, homework, attention, previous]

scaler_main = StandardScaler()
X = scaler_main.fit_transform(df[features])

model_perf = LinearRegression().fit(X, df["performance_index"])
model_pass = LogisticRegression(max_iter=1000).fit(X, df["pass_fail"])
model_risk = LogisticRegression(max_iter=1000).fit(X, df["at_risk"])

df["perf_cat"] = pd.cut(df["average_marks"], [0, 50, 75, 100], labels=[0, 1, 2])
model_knn = KNeighborsClassifier(7).fit(X, df["perf_cat"])

scaler_eng = StandardScaler()
X_eng = scaler_eng.fit_transform(df[[attendance, homework, attention]])
model_eng = LinearRegression().fit(X_eng, df["engagement_score"])

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df[["average_marks", "engagement_score"]])
model_kmeans = KMeans(4, random_state=42).fit(X_cluster)
df["cluster"] = model_kmeans.labels_

# =====================================================
# METRICS
# =====================================================
conf_reg = round(r2_score(df["performance_index"], model_perf.predict(X)) * 100, 2)
conf_pass = round(accuracy_score(df["pass_fail"], model_pass.predict(X)) * 100, 2)
conf_risk = round(accuracy_score(df["at_risk"], model_risk.predict(X)) * 100, 2)

analytics_data = {
    "pass_fail": [
        int(df["pass_fail"].sum()),
        int(len(df) - df["pass_fail"].sum())
    ],
    "cluster_marks": df.groupby("cluster")["average_marks"].mean().round(2).tolist(),
    "cluster_eng": df.groupby("cluster")["engagement_score"].mean().round(2).tolist(),
    "model_perf": [conf_reg, conf_pass, conf_risk]
}

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

# =====================================================
# MAIN PAGE HTML
# =====================================================
MAIN_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>SUDHIR TUTORIALS - Edupredict AI</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{background:#0f2027;color:white;}
.card{border-radius:18px;}
footer{text-align:center;color:#aaa;margin-top:50px;}
</style>
</head>
<body class="container py-5">

<h1 class="text-center">SUDHIR TUTORIALS - Edupredict AI</h1>
<p class="text-center text-muted">Intelligent Student Prediction System</p>

<form method="POST" class="mt-4">

<select name="ptype" class="form-select mb-3" required>
<option value="">Select Prediction</option>
<option value="performance">Performance Index</option>
<option value="pass">Pass Probability</option>
<option value="risk">At-Risk Probability</option>
<option value="category">Performance Category</option>
<option value="engagement">Engagement Score</option>
<option value="cluster">Student Cluster</option>
</select>

<div class="row">
{% for f in ["science","maths","attendance","homework","attention","previous"] %}
<div class="col-md-4 mb-3">
<input name="{{f}}" class="form-control" placeholder="{{f|capitalize}}" required>
</div>
{% endfor %}
</div>

<button class="btn btn-warning w-100">Predict</button>
</form>

{% if output %}
<hr>
<div class="card bg-dark p-4 mt-4">
<h4>{{output.title}}</h4>
<h2 class="text-warning">{{output.value}}</h2>
{% if output.conf %}
<p class="text-muted">{{output.conf}}</p>
{% endif %}
<a href="/analytics" class="btn btn-outline-info mt-3">View Analytics Dashboard</a>
</div>
{% endif %}

<footer>
© 2025 SUDHIR TUTORIALS - Edupredict AI
</footer>

</body>
</html>
"""

# =====================================================
# ANALYTICS HTML
# =====================================================
ANALYTICS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>SUDHIR TUTORIALS - Edupredict AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{
background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);
color:white;
}
.card{border-radius:18px;}
footer{text-align:center;color:#ccc;margin-top:40px;}
</style>
</head>

<body class="container py-5">

<h1 class="text-center">SUDHIR TUTORIALS - Edupredict AI</h1>
<h4 class="text-center text-muted mb-4">Analytics Dashboard</h4>

<div class="text-center mb-4">
<a href="/" class="btn btn-outline-light">⬅ Back to Prediction</a>
</div>

<div class="row">
<div class="col-md-6 card p-3 mb-4"><canvas id="pieChart"></canvas></div>
<div class="col-md-6 card p-3 mb-4"><canvas id="modelChart"></canvas></div>
</div>

<div class="row">
<div class="col-md-6 card p-3 mb-4"><canvas id="marksChart"></canvas></div>
<div class="col-md-6 card p-3 mb-4"><canvas id="engChart"></canvas></div>
</div>

<footer>
© 2025 SUDHIR TUTORIALS - Edupredict AI
</footer>

<script>
const data = {{ data | safe }};

new Chart(document.getElementById("pieChart"),{
type:"pie",
data:{labels:["Pass","Fail"],
datasets:[{data:data.pass_fail,backgroundColor:["#28a745","#dc3545"]}]}
});

new Chart(document.getElementById("modelChart"),{
type:"bar",
data:{labels:["Regression","Pass Model","Risk Model"],
datasets:[{label:"Model Performance (%)",
data:data.model_perf,
backgroundColor:["#17a2b8","#ffc107","#fd7e14"]}]}
});

new Chart(document.getElementById("marksChart"),{
type:"bar",
data:{labels:["Cluster 0","Cluster 1","Cluster 2","Cluster 3"],
datasets:[{label:"Average Marks",
data:data.cluster_marks,
backgroundColor:"#6f42c1"}]}
});

new Chart(document.getElementById("engChart"),{
type:"bar",
data:{labels:["Cluster 0","Cluster 1","Cluster 2","Cluster 3"],
datasets:[{label:"Engagement Score",
data:data.cluster_eng,
backgroundColor:"#20c997"}]}
});
</script>

</body>
</html>
"""

# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET","POST"])
def home():
    output = None
    if request.method == "POST":
        try:
            t = request.form.get("ptype")
            s = float(request.form["science"])
            m = float(request.form["maths"])
            a = float(request.form["attendance"])
            h = float(request.form["homework"])
            c = float(request.form["attention"])
            p = float(request.form["previous"])

            X_main = scaler_main.transform([[s, m, a, h, c, p]])

            if t == "performance":
                val = model_perf.predict(X_main)[0]
                output = {"title": "Performance Index", "value": round(val, 2), "conf": f"Model R²: {conf_reg}%"}

            elif t == "pass":
                prob = model_pass.predict_proba(X_main)[0][1] * 100
                output = {"title": "Pass Probability", "value": f"{round(prob,2)}%", "conf": f"Accuracy: {conf_pass}%"}

            elif t == "risk":
                prob = model_risk.predict_proba(X_main)[0][1] * 100
                output = {"title": "At-Risk Probability", "value": f"{round(prob,2)}%", "conf": f"Accuracy: {conf_risk}%"}

            elif t == "category":
                cat = int(model_knn.predict(X_main)[0])
                output = {"title": "Performance Category", "value": ["Weak","Average","Excellent"][cat], "conf": None}

            elif t == "engagement":
                val = model_eng.predict(scaler_eng.transform([[a, h, c]]))[0]
                output = {"title": "Engagement Score", "value": round(val,2), "conf": None}

            elif t == "cluster":
                eng = model_eng.predict(scaler_eng.transform([[a, h, c]]))[0]
                cl = model_kmeans.predict(scaler_cluster.transform([[ (s+m)/2, eng ]]))[0]
                output = {"title": "Student Cluster", "value": cl, "conf": None}

        except:
            output = {"title": "Error", "value": "Invalid Input", "conf": None}

    return render_template_string(MAIN_HTML, output=output)

@app.route("/analytics")
def analytics():
    return render_template_string(
        ANALYTICS_HTML,
        data=json.dumps(analytics_data)
    )

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



