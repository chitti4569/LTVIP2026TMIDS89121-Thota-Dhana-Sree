import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn
import sklearn
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from sklearn import svm
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from xgboost import XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import LinearSVC

data = pd.read_csv('data/Weather.csv');
data.head();

data.describe();

data.info();

data.shape

data.isnull().sum();

msno.matrix(data, color=(0.55, 0.255, 0.225), fontsize=16)
plt.show();

data_cat = data[['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']]

data.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], inplace=True)
data.drop(columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], inplace=True)

data['MinTemp'] = data['MinTemp'].fillna(data['MinTemp'].mean())
data['MaxTemp'] = data['MaxTemp'].fillna(data['MaxTemp'].mean())
data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].mean())
data['WindSpeed9am'] = data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean())
data['WindSpeed3pm'] = data['WindSpeed3pm'].fillna(data['WindSpeed3pm'].mean())
data['Humidity9am'] = data['Humidity9am'].fillna(data['Humidity9am'].mean())
data['Humidity3pm'] = data['Humidity3pm'].fillna(data['Humidity3pm'].mean())
data['Pressure9am'] = data['Pressure9am'].fillna(data['Pressure9am'].mean())
data['Pressure3pm'] = data['Pressure3pm'].fillna(data['Pressure3pm'].mean())
data['Temp9am'] = data['Temp9am'].fillna(data['Temp9am'].mean())
data['Temp3pm'] = data['Temp3pm'].fillna(data['Temp3pm'].mean())

print(data);

cat_names = data_cat.columns

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

data_cat = imp_mode.fit_transform(data_cat)

data_cat = pd.DataFrame(data_cat, columns=cat_names)

data = pd.concat([data, data_cat], axis=1)

print(data)

# correlation matrix
cor = data.corr(numeric_only=True)

# heatmap (same style as image)
sns.heatmap(
    data=cor,
    xticklabels=cor.columns.values,
    yticklabels=cor.columns.values
)

plt.show()

# ---------- HEATMAP ----------
cor = data.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cor,
    xticklabels=cor.columns.values,
    yticklabels=cor.columns.values
)
plt.show()

pair_cols = [
    'MinTemp',
    'MaxTemp',
    'Humidity9am',
    'Humidity3pm',
    'Pressure9am',
    'Pressure3pm',
    'RainToday'
]

pair_data = data[pair_cols].dropna()

pair_data = pair_data.sample(n=500, random_state=42)

sns.set(style="white", font_scale=0.9)

sns.pairplot(
    pair_data,
    hue='RainToday',
    diag_kind='kde',
    palette=['purple', 'orange'],
    markers=['^', 'v'],
    plot_kws={'alpha': 0.6, 's': 25},
    diag_kws={'fill': True}
)

plt.show()

data.boxplot();
plt.show();

x = data["MinTemp"]
y = data["Rainfall"]

# -----------------------------
# Create joint-plot layout
# -----------------------------
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

# Main scatter plot
ax_scatter = fig.add_subplot(grid[1:4, 0:3])
ax_scatter.scatter(x, y, alpha=0.6)
ax_scatter.set_xlabel("MinTemp")
ax_scatter.set_ylabel("Rainfall")
ax_scatter.set_title("MinTemp vs Rainfall")

# Top histogram (MinTemp)
ax_histx = fig.add_subplot(grid[0, 0:3], sharex=ax_scatter)
ax_histx.hist(x, bins=30)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histx.set_ylabel("Frequency")

# Right histogram (Rainfall)
ax_histy = fig.add_subplot(grid[1:4, 3], sharey=ax_scatter)
ax_histy.hist(y, bins=30, orientation="horizontal")
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlabel("Frequency")

# -----------------------------
# Display plot
# -----------------------------
plt.show()

data_no = data[data["RainTomorrow"] == "No"]
data_yes = data[data["RainTomorrow"] == "Yes"]

# Create figure layout
fig = plt.figure(figsize=(9, 9))
grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

# Scatter plot
ax_scatter = fig.add_subplot(grid[1:4, 0:3])
ax_scatter.scatter(data_no["MaxTemp"], data_no["Rainfall"], alpha=0.6, label="No")
ax_scatter.scatter(data_yes["MaxTemp"], data_yes["Rainfall"], alpha=0.6, label="Yes")

ax_scatter.set_xlabel("MaxTemp")
ax_scatter.set_ylabel("Rainfall")
ax_scatter.set_title("MaxTemp vs Rainfall (RainTomorrow)")
ax_scatter.legend()

# Top histogram
ax_histx = fig.add_subplot(grid[0, 0:3], sharex=ax_scatter)
ax_histx.hist(data_no["MaxTemp"], bins=30, alpha=0.6)
ax_histx.hist(data_yes["MaxTemp"], bins=30, alpha=0.6)
ax_histx.tick_params(axis="x", labelbottom=False)

# Right histogram
ax_histy = fig.add_subplot(grid[1:4, 3], sharey=ax_scatter)
ax_histy.hist(data_no["Rainfall"], bins=30, orientation="horizontal", alpha=0.6)
ax_histy.hist(data_yes["Rainfall"], bins=30, orientation="horizontal", alpha=0.6)
ax_histy.tick_params(axis="y", labelleft=False)

plt.show()

counts = data["RainTomorrow"].value_counts()

# Plot
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values)

plt.xlabel("RainTomorrow")
plt.ylabel("Count")
plt.title("RainTomorrow Distribution")

plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(
    data["MaxTemp"],
    data["Rainfall"],
    alpha=0.6
)

plt.xlabel("MaxTemp")
plt.ylabel("Rainfall")
plt.title("MaxTemp vs Rainfall")

plt.show()

plt.figure(figsize=(7, 5))
plt.hist(
    data["MinTemp"],
    bins=50
)

plt.xlabel("MinTemp")
plt.ylabel("Count")
plt.title("Distribution of MinTemp")

plt.show()

# =============================
# BASIC CLEANING
# =============================
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# DROP useless / auto-generated columns
data.drop(columns=['@dropdown'], inplace=True, errors='ignore')

# DROP columns with too many missing values
drop_cols = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
data.drop(columns=drop_cols, inplace=True, errors='ignore')

# =============================
# HANDLE DATE COLUMN
# =============================
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day
    data.drop(columns=['Date'], inplace=True)

# =============================
# REMOVE NaN FROM TARGET (CRITICAL FIX)
# =============================
data = data.dropna(subset=['RainTomorrow'])

# =============================
# SPLIT X AND y
# =============================
y = data['RainTomorrow']
X = data.drop(columns=['RainTomorrow'])

# =============================
# IDENTIFY COLUMN TYPES (FUTURE-SAFE)
# =============================
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# =============================
# NUMERIC PIPELINE
# =============================
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("variance", VarianceThreshold(threshold=0)),
    ("scaler", StandardScaler())
])

# =============================
# CATEGORICAL PIPELINE (SPARSE + VERSION SAFE)
# =============================
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=True
    ))
])

# =============================
# COLUMN TRANSFORMER (MEMORY SAFE)
# =============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ],
    sparse_threshold=0.3
)

# =============================
# APPLY PREPROCESSING
# =============================
X_processed = preprocessor.fit_transform(X)

print("Final feature matrix shape:", X_processed.shape)
print("Matrix type:", type(X_processed))

# =============================
# SPLIT X AND y
# =============================
y = data['RainTomorrow'].map({'No': 0, 'Yes': 1})
X = data.drop(columns=['RainTomorrow'])

# =============================
# TRAIN–TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

XGBoost = XGBRFClassifier(n_estimators=50, random_state=42)
Rand_forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
svm = LinearSVC(max_iter=20000, tol=1e-3, dual=False)

Dtree = DecisionTreeClassifier(random_state=42)
GBM = GradientBoostingClassifier(random_state=42)
log = LogisticRegression(max_iter=1000)

xgb_model = XGBoost
rf_model  = Rand_forest
svm_model = svm
dt_model  = Dtree
gb_model  = GBM
log_model = log


XGBoost.fit(X_train, y_train)
Rand_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)
Dtree.fit(X_train, y_train)
GBM.fit(X_train, y_train)
log.fit(X_train, y_train)


print(y_train.unique())

p1 = XGBoost.predict(X_test)
p2 = Rand_forest.predict(X_test)
p3 = svm.predict(X_test)
p4 = Dtree.predict(X_test)
p5 = GBM.predict(X_test)
p6 = log.predict(X_test)

# =============================
# TEST ACCURACY
# =============================
print("xgboost:", metrics.accuracy_score(y_test, p1))
print("Rand_forest:", metrics.accuracy_score(y_test, p2))
print("svm:", metrics.accuracy_score(y_test, p3))
print("Dtree:", metrics.accuracy_score(y_test, p4))
print("GBM:", metrics.accuracy_score(y_test, p5))
print("log:", metrics.accuracy_score(y_test, p6))

t1 = XGBoost.predict(X_test)
t2 = Rand_forest.predict(X_test)
t3 = svm.predict(X_test)
t4 = Dtree.predict(X_test)
t5 = GBM.predict(X_test)
t6 = log.predict(X_test)

print("xgboost:", metrics.accuracy_score(y_test, t1))
print("Rand_forest:", metrics.accuracy_score(y_test, t2))
print("svm:", metrics.accuracy_score(y_test, t3))
print("Dtree:", metrics.accuracy_score(y_test, t4))
print("GBM:", metrics.accuracy_score(y_test, t5))
print("log:", metrics.accuracy_score(y_test, t6))


y_pred = Rand_forest.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, alpha=0.3)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Actuals", fontsize=18)
plt.title("Confusion Matrix (Random Forest)", fontsize=18)
plt.show()

# Probabilities for ROC
y_prob = Rand_forest.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', label='Random Guess')
plt.fill_between(fpr, tpr, alpha=0.3)

plt.text(0.6, 0.2, f'AUC = {auc:.4f}',
         fontsize=12, weight='bold')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend()
plt.show()

# 1. Save the actual model you trained
# In your script, this was named 'Rand_forest'
pickle.dump(Rand_forest, open('rainfall.pkl', 'wb'))

# 2. Save the preprocessor
# This replaces 'le', 'imp_mode', and 'sc' because your ColumnTransformer
# already contains the Imputer, Scaler, and OneHotEncoder combined.
pickle.dump(preprocessor, open('encoder.pkl', 'wb'))

# 3. Save the categorical imputer (Optional)
# If you specifically need the categorical imputer you named 'imp_mode'
pickle.dump(imp_mode, open('impter.pkl', 'wb'))

# This object contains your scaling logic
pickle.dump(preprocessor, open('scale.pkl', 'wb'))
