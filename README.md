# 🔬 Smart Auto-Analytics and Insights Generator

Upload **any** CSV or Excel dataset — sales, students, employees, hospitals, cricket, anything. This tool automatically profiles your data, generates up to **8 plain-English insights**, creates the right charts, detects anomalies using dual-mode detection, and answers **15+ types of plain-English questions** instantly. No code required.

---

## 🚀 Live Demo

> Run locally — see setup instructions below.

---

## 📌 What It Does

| Tab | Feature | Description |
|---|---|---|
| 📋 | Data Profile | Column-by-column breakdown — types, missing %, mean, median, min, max |
| 💡 | Smart Insights | Up to 8 auto-generated plain-English observations backed by statistics |
| 📊 | Auto Charts | Correct chart type selected automatically per column |
| 🔍 | Anomaly Detection | Dual-mode: IQR outliers OR MAD-based ranking for clean data |
| 💬 | Ask Your Data | 15+ plain-English question types answered instantly — no code needed |

---

## 💡 Smart Insights Engine

The tool automatically detects and writes 7 insight types, returning up to 8 total:

| Insight Type | Example |
|---|---|
| 📋 Dataset Overview | "150 rows, 8 columns, 0% missing, 0 duplicates" |
| ⚠️ High Missing Values | "42.0% of values in 'email' are missing — needs attention before modeling" |
| 🔗 Correlation | "'midterm_score' and 'final_score' are strongly positively correlated (r = 0.82)" |
| 📍 Outliers | "12 rows (8.0%) in 'sales' are statistical outliers by IQR method" |
| 📊 Category Imbalance | "'Mobile' accounts for 72% of records in 'platform'" |
| 📈 High Variability | "'salary' has a CV of 61% — wide spread worth investigating" |
| 🏆 Range Highlight | "Highest salary: 120,000. Lowest: 25,000. Top 10% average: 105,000" |
| ✅ Clean Data | "No missing values and no duplicates — dataset ready for analysis" |

ID-like columns (ending in `_id`, `_no`) are automatically excluded from variability analysis to prevent meaningless flags.

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

## 💬 Ask Your Data — 15+ Supported Question Types

The query engine uses a **4-pass fuzzy column matching system** — handles exact matches, underscore-normalized names, no-space names, and typo-tolerant fuzzy matching. Type naturally, even with typos.

```
"how many rows"                     → The dataset has 200 rows.
"what columns are there"            → Columns (7): date, region, sales, ...
"what is average sales"             → Average: 51,029.31 | Median: 48,200.00
"highest sales"                     → Highest sales: 107,790.97 (Row 57)
"lowest salary"                     → Lowest salary: 25,000.00 (Row 12)
"total revenue"                     → Total revenue: 4,250,000.00
"show top 10 region"                → Top 10 in region: North: 54, East: 52 ...
"which region has highest sales"    → Region with highest avg sales: North (72,400.00)
"how many missing values"           → Missing values by column: ...
"what is the correlation"           → Strong correlations (|r| > 0.5): ...
"describe sales"                    → count, mean, std, min, 25%, 50%, 75%, max
"standard deviation of salary"      → Std Dev: 18,450.00 | Variance: 340,403,500
"skewness of sales"                 → 1.243 — right-skewed (few very high values pull avg up)
"is salary normally distributed"    → Shapiro-Wilk p = 0.032 → not normal (p ≤ 0.05)
"outliers in sales"                 → 12 of 200 rows (6.0%) outside IQR bounds
```

---

## 📊 Auto-Visualization Logic

| Column Type | Chart Generated |
|---|---|
| Numeric (continuous) | Histogram — shows distribution |
| Text (≤ 20 unique values) | Horizontal bar chart — top 10 categories |
| Multiple numeric columns | Correlation heatmap (up to 8 columns) |
| Two most correlated columns | Scatter plot with correlation value in title |
| Columns with missing values | Missing values percentage bar chart |

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
| SciPy | Shapiro-Wilk normality test |
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
scipy
openpyxl
```

Save as `requirements.txt` in the project folder.

---

## 📸 Screenshots

**Smart Insights — Auto-Generated Observations**
<img width="7684" height="4322" alt="insights" src="https://github.com/user-attachments/assets/7588f433-d313-4d4f-8e6a-389aa5fa1dcb" />

**Auto Charts — Correlation Heatmap**
<img width="7684" height="4322" alt="auto_charts" src="https://github.com/user-attachments/assets/ab81c944-a264-485d-b614-49b2610086aa" />

**Anomaly Detection — Most Unusual Rows**
<img width="7684" height="4322" alt="anomaly_detection" src="https://github.com/user-attachments/assets/6c7ac5ef-3b40-4a4b-ac0d-deb82c580699" />

**Ask Your Data — Plain English Queries**
<img width="7684" height="4322" alt="ask_your_data" src="https://github.com/user-attachments/assets/3bf91db7-1d68-450e-ad22-b50641dac99e" />


---

## 👤 Developer

**Kartikeya Warhade**
[LinkedIn](https://linkedin.com/in/kartikeya-warhade) · [GitHub](https://github.com/KartikeyaWarhade2002)
