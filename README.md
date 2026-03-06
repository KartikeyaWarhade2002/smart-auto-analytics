# 🔬 Smart Auto-Analytics and Insights Generator

Upload **any** CSV or Excel dataset — sales, students, employees, hospitals, cricket, anything. This tool automatically understands your data, generates up to **8 plain-English insights**, creates the right charts, detects anomalies, and answers your questions in plain English. No code required.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 📌 What It Does

| Tab | Feature | Description |
|---|---|---|
| 📋 | Data Profile | Column-by-column breakdown — types, missing values, stats |
| 💡 | Smart Insights | Up to 8 auto-generated plain-English observations |
| 📊 | Auto Charts | Right chart type selected automatically per column |
| 🔍 | Anomaly Detection | Dual-mode: IQR outliers OR MAD-based ranking |
| 💬 | Ask Your Data | Plain-English questions answered instantly |

---

## 💡 Smart Insights Engine

The tool automatically finds and writes observations about your data:

| Insight Type | Example |
|---|---|
| 📋 Dataset Overview | "150 rows, 8 columns, 0% missing, 0 duplicates" |
| 🔗 Correlation | "midterm_score and final_score are strongly positively correlated (r = 0.82)" |
| 📍 Outliers | "12 rows (8.0%) in sales are statistical outliers by IQR method" |
| 📊 Category Imbalance | "'Mobile' accounts for 72% of records in platform" |
| 📈 High Variability | "salary has CV of 61% — wide spread worth investigating" |
| 🏆 Range Highlight | "Highest salary: 120,000. Lowest: 25,000. Top 10% average: 105,000" |
| ✅ Clean Data | "No missing values and no duplicates — dataset ready for analysis" |

---

## 🔍 Dual-Mode Anomaly Detection

The tool uses **two methods** depending on your data:

**Mode 1 — IQR Outlier Detection** (when hard outliers exist)
- Flags rows where values fall outside Q1 − 1.5×IQR or Q3 + 1.5×IQR
- Shows exact count of columns violated per row
- Includes anomaly score distribution chart

**Mode 2 — MAD-Based Ranking** (when data is clean and uniform)
- Uses Median Absolute Deviation — more robust than standard deviation
- Ranks every row by deviation from median across all numeric columns
- Always shows the most statistically unusual rows even without hard outliers
- Binary columns (0/1 flags) are automatically excluded from both methods

---

## 💬 Ask Your Data — Supported Questions

```
"how many rows"                   → The dataset has 200 rows.
"what columns are there"          → Columns (7): date, region, sales, ...
"what is average sales"           → Average sales: 51,029.31
"highest sales"                   → Highest sales: 107,790.97 (Row 57)
"lowest salary"                   → Lowest salary: 25,000.00 (Row 12)
"show top 10 region"              → Top 10 in region: North: 54, East: 52 ...
"how many missing values"         → Missing values by column: ...
"what is the correlation"         → Strong correlations (|r| > 0.5): ...
"describe sales"                  → count, mean, std, min, 25%, 50%, 75%, max
"total revenue"                   → Total revenue: 4,250,000.00
```

---

## 📊 Auto-Visualization Logic

| Column Type | Chart Generated |
|---|---|
| Numeric (continuous) | Histogram — shows distribution |
| Text (≤ 20 unique values) | Horizontal bar chart — top 10 categories |
| Multiple numeric columns | Correlation heatmap |
| Two most correlated columns | Scatter plot with correlation value |
| Columns with missing values | Missing values bar chart |

---

## 🗂️ Sample Datasets (built-in)

No upload needed — try with built-in samples:

| Dataset | Rows | Description |
|---|---|---|
| Retail Sales | 200 | Region, category, sales, units, customers |
| Student Performance | 150 | Scores, attendance, CGPA — realistic correlations |
| Employee Data | 100 | Salary vs experience, performance vs projects |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading, profiling, cleaning |
| NumPy | Statistical calculations, MAD scoring |
| Matplotlib | Histograms, bar charts, scatter plots |
| Seaborn | Correlation heatmap |
| Streamlit | Interactive web dashboard |

---

## 📁 Project Structure

```
smart-auto-analytics/
│
├── app.py               # Full Streamlit app (all modules included)
├── requirements.txt     # All dependencies
└── README.md
```

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/KartikeyaWarhade2002/smart-auto-analytics.git
cd smart-auto-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
openpyxl
```

Save as `requirements.txt` in the project folder.

---

## 👤 Developer

**Kartikeya Babaraoji Warhade**
[LinkedIn](https://linkedin.com/in/kartikeya-warhade) · [GitHub](https://github.com/KartikeyaWarhade2002)
