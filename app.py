# =============================================================================
# PROJECT 3: Smart Auto-Analytics and Insights Generator
# Developer: Kartikeya Warhade
#
# WHAT IT DOES:
# Upload any dataset — sales, students, employees, hospitals, cricket, anything.
# The tool automatically understands your data, finds the 5 most interesting
# patterns, generates the right charts, detects anomalies, and lets you
# ask plain-English questions about your data.
#
# TECH STACK (simple and fully explainable):
#   Pandas      — data loading, cleaning, profiling
#   NumPy       — numerical calculations
#   Matplotlib  — charts and visualizations
#   Seaborn     — correlation heatmap
#   Streamlit   — interactive web dashboard
#
# NO deep learning. NO black boxes. Every feature explained in one sentence.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime


# =============================================================================
# SECTION 1 — FILE LOADER
# Handles CSV and Excel. Cleans column names automatically.
# =============================================================================

def load_file(uploaded_file):
    """
    Loads CSV or Excel file into a Pandas DataFrame.
    Cleans column names by stripping whitespace.
    Returns (dataframe, error_message)
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            # Try UTF-8 first, fall back to latin-1 for special characters
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload CSV or Excel."

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Drop completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        return df, None

    except Exception as e:
        return None, f"Could not read file: {str(e)}"


# =============================================================================
# SECTION 2 — DATA PROFILER
# Scans every column and records its characteristics.
# This is the DATA ANALYTICS layer — understand before analyzing.
# =============================================================================

def profile_data(df):
    """
    Profiles every column in the dataset.

    For each column records:
    - Data type (number, text, date)
    - Missing value count and percentage
    - Number of unique values
    - For numbers: mean, median, min, max, std deviation
    - For text: most common value and its frequency

    WHY profile first?
    Before any analysis you must understand what you have.
    A doctor does a full checkup before prescribing medicine.
    A data scientist profiles the data before building any model.
    """
    profile = []

    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        missing = int(col_data.isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1)
        unique = int(col_data.nunique())

        row = {
            'Column': col,
            'Type': 'Numeric' if dtype in ['int64', 'float64'] else
                    'Date' if 'datetime' in dtype else 'Text',
            'Missing': f"{missing} ({missing_pct}%)",
            'Unique Values': unique,
            'Details': ''
        }

        if dtype in ['int64', 'float64']:
            clean = col_data.dropna()
            if len(clean) > 0:
                row['Details'] = (
                    f"Mean: {clean.mean():.2f} | "
                    f"Median: {clean.median():.2f} | "
                    f"Min: {clean.min():.2f} | "
                    f"Max: {clean.max():.2f}"
                )
        else:
            if len(col_data.dropna()) > 0:
                top_val = col_data.value_counts().index[0]
                top_count = col_data.value_counts().iloc[0]
                row['Details'] = f"Most common: '{top_val}' ({top_count} times)"

        profile.append(row)

    return pd.DataFrame(profile)


# =============================================================================
# SECTION 3 — SMART INSIGHT GENERATOR
# Automatically finds the 5 most interesting things in the dataset.
# Writes them as plain-English sentences.
#
# HOW IT WORKS:
# Scans correlations, distributions, missing values, category imbalances,
# and outliers. Picks the most statistically significant findings.
# Converts each finding into a readable sentence.
#
# WHY THIS IS IMPRESSIVE:
# Most tools just show numbers. This tool interprets numbers and writes
# sentences about them — the way a senior analyst would in a report.
# =============================================================================

def generate_insights(df):
    """
    Automatically generates up to 8 plain-English insights from the data.
    Each insight is a specific, data-backed observation.

    FIXES:
    - ID columns excluded (student_id, employee_id never flagged as high variability)
    - Correlation threshold lowered 0.7 → 0.5 (catches more real patterns)
    - Added Range Highlight and Clean Data insights so count is always 5-8
    - Dataset Overview moved to top and always shown first
    """
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols    = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude ID-like columns from numeric analysis — they are meaningless metrics
    def is_id_col(col):
        n = col.lower()
        if n.endswith('id') or n == 'id' or n.endswith('_id') or n.endswith('_no') or n == 'index':
            data = df[col].dropna()
            if data.dtype in ['int64','float64']:
                return data.nunique() >= len(data) * 0.95  # nearly all unique = ID
        return False

    analysis_numeric = [c for c in numeric_cols if not is_id_col(c)]

    # ── ALWAYS FIRST: DATASET OVERVIEW ───────────────────────────────────────
    total_missing   = int(df.isnull().sum().sum())
    missing_pct_all = total_missing / (df.shape[0] * df.shape[1]) * 100
    dup_count       = int(df.duplicated().sum())
    insights.append({
        'type': 'info', 'icon': '📋',
        'title': 'Dataset Overview',
        'detail': (
            f"Dataset has {len(df):,} rows and {len(df.columns)} columns. "
            f"Overall missing data: {missing_pct_all:.1f}%. "
            f"Duplicate rows: {dup_count}. "
            f"Numeric columns: {len(numeric_cols)}. "
            f"Text columns: {len(text_cols)}."
        )
    })

    # ── INSIGHT TYPE 1: HIGH MISSING VALUES ──────────────────────────────────
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 20:
            insights.append({
                'type': 'warning', 'icon': '⚠️',
                'title': f'High Missing Values in "{col}"',
                'detail': (
                    f"{missing_pct:.1f}% of values in '{col}' are missing. "
                    f"This column needs attention before any analysis or ML model training."
                )
            })

    # ── INSIGHT TYPE 2: CORRELATIONS (threshold 0.5, was 0.7) ────────────────
    if len(analysis_numeric) >= 2:
        corr_matrix = df[analysis_numeric].corr()
        found = []
        for i in range(len(analysis_numeric)):
            for j in range(i+1, len(analysis_numeric)):
                v    = corr_matrix.iloc[i, j]
                ca, cb = analysis_numeric[i], analysis_numeric[j]
                if abs(v) > 0.5:
                    direction = "positively" if v > 0 else "negatively"
                    strength  = "strongly" if abs(v) > 0.75 else "moderately"
                    found.append((abs(v), {
                        'type': 'insight', 'icon': '🔗',
                        'title': f'Relationship: "{ca}" ↔ "{cb}"',
                        'detail': (
                            f"'{ca}' and '{cb}' are {strength} {direction} correlated "
                            f"(r = {v:.2f}). "
                            + ("When one increases, the other tends to increase." if v > 0
                               else "When one increases, the other tends to decrease.")
                        )
                    }))
        for _, ins in sorted(found, reverse=True)[:2]:
            insights.append(ins)

    # ── INSIGHT TYPE 3: OUTLIERS ──────────────────────────────────────────────
    for col in analysis_numeric[:5]:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        out_count = int(((clean < Q1-1.5*IQR) | (clean > Q3+1.5*IQR)).sum())
        out_pct   = out_count / len(clean) * 100
        if out_pct > 3:
            insights.append({
                'type': 'warning', 'icon': '📍',
                'title': f'Outliers Detected in "{col}"',
                'detail': (
                    f"{out_count} rows ({out_pct:.1f}%) in '{col}' are statistical outliers "
                    f"(IQR method). Values range from {clean.min():.2f} to {clean.max():.2f}. "
                    f"Outliers may be data entry errors or genuinely exceptional cases."
                )
            })

    # ── INSIGHT TYPE 4: CATEGORY IMBALANCE ───────────────────────────────────
    for col in text_cols[:4]:
        if df[col].nunique() < 2 or df[col].nunique() > 30:
            continue
        vc      = df[col].value_counts()
        top_pct = vc.iloc[0] / len(df) * 100
        if top_pct > 60:
            insights.append({
                'type': 'insight', 'icon': '📊',
                'title': f'Dominant Category in "{col}"',
                'detail': (
                    f"'{vc.index[0]}' accounts for {top_pct:.1f}% of records in '{col}'. "
                    f"This imbalance could affect statistical analysis and ML model training."
                )
            })

    # ── INSIGHT TYPE 5: HIGH VARIABILITY (ID cols excluded) ──────────────────
    for col in analysis_numeric[:4]:
        clean = df[col].dropna()
        if len(clean) < 5:
            continue
        mean_v = clean.mean()
        cv = (clean.std() / mean_v * 100) if mean_v != 0 else 0
        if cv > 50:
            insights.append({
                'type': 'insight', 'icon': '📈',
                'title': f'High Variability in "{col}"',
                'detail': (
                    f"'{col}' has a coefficient of variation of {cv:.1f}%. "
                    f"Mean is {mean_v:,.2f} but values range from {clean.min():,.2f} "
                    f"to {clean.max():,.2f}. Wide spread worth investigating."
                )
            })

    # ── INSIGHT TYPE 6: RANGE HIGHLIGHT — always adds one more insight ────────
    if analysis_numeric and len(insights) < 6:
        best_col = max(analysis_numeric,
                       key=lambda c: df[c].std()/df[c].mean() if df[c].mean() != 0 else 0)
        clean = df[best_col].dropna()
        if len(clean) >= 5:
            top10_avg = clean.quantile(0.9)
            insights.append({
                'type': 'insight', 'icon': '🏆',
                'title': f'Range Highlight: "{best_col}"',
                'detail': (
                    f"Highest value in '{best_col}': {clean.max():,.2f}. "
                    f"Lowest: {clean.min():,.2f}. "
                    f"Mean: {clean.mean():,.2f}. "
                    f"Top 10% of rows average {top10_avg:,.2f} in this column."
                )
            })

    # ── INSIGHT TYPE 7: CLEAN DATA CONFIRMATION ───────────────────────────────
    if total_missing == 0 and dup_count == 0:
        insights.append({
            'type': 'info', 'icon': '✅',
            'title': 'Clean Dataset Confirmed',
            'detail': (
                "No missing values and no duplicate rows found. "
                "This dataset is ready for direct analysis or model training."
            )
        })

    return insights[:8]


# =============================================================================
# SECTION 4 — AUTO VISUALIZATION ENGINE
# Generates the right chart type for each column automatically.
#
# LOGIC:
# - Numeric column with many unique values → Histogram (distribution)
# - Numeric column with few unique values → Bar chart (count)
# - Text column with ≤15 unique values → Bar chart (frequency)
# - Two numeric columns → Scatter plot (relationship)
# - Date column → Line chart (trend)
#
# WHY AUTO-SELECTION:
# A histogram for continuous data, a bar chart for categories.
# The wrong chart type misleads the viewer.
# This tool always picks the statistically correct chart.
# =============================================================================

def create_auto_visualizations(df):
    """
    Generates up to 6 automatic visualizations based on column types.
    Returns a list of matplotlib figures.
    """
    figures = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [c for c in df.select_dtypes(include=['object']).columns
                 if df[c].nunique() <= 20]

    # ── CHART 1: DISTRIBUTIONS OF NUMERIC COLUMNS ────────────────────────────
    if len(numeric_cols) >= 1:
        n_cols = min(len(numeric_cols), 4)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(numeric_cols[:4]):
            clean = df[col].dropna()
            axes[i].hist(clean, bins=20, color='#1565C0', edgecolor='white', alpha=0.85)
            axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        fig.suptitle('Distribution of Numeric Columns', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        figures.append(('Distributions', fig))

    # ── CHART 2: TOP CATEGORIES IN TEXT COLUMNS ───────────────────────────────
    if len(text_cols) >= 1:
        n_cols = min(len(text_cols), 2)
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(text_cols[:2]):
            counts = df[col].value_counts().head(10)
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))
            axes[i].barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
            axes[i].set_title(f'Top Values — {col}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Count')
            axes[i].grid(True, alpha=0.3, axis='x')

        fig.suptitle('Category Distributions', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        figures.append(('Categories', fig))

    # ── CHART 3: CORRELATION HEATMAP ──────────────────────────────────────────
    # Shows relationships between ALL numeric columns at once
    if len(numeric_cols) >= 2:
        cols_for_corr = numeric_cols[:8]  # max 8 for readability
        corr = df[cols_for_corr].corr()

        fig, ax = plt.subplots(figsize=(max(6, len(cols_for_corr)), max(5, len(cols_for_corr)-1)))
        sns.heatmap(
            corr, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0,
            square=True, ax=ax,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9}
        )
        ax.set_title('Correlation Matrix — How Columns Relate to Each Other',
                     fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        figures.append(('Correlations', fig))

    # ── CHART 4: SCATTER PLOT — TOP 2 CORRELATED COLUMNS ─────────────────────
    if len(numeric_cols) >= 2:
        # Find the two most correlated columns
        corr_matrix = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_idx = corr_matrix.stack().idxmax()
        col_x, col_y = max_idx[0], max_idx[1]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[col_x], df[col_y],
                  alpha=0.5, color='#1565C0', s=20, edgecolors='none')
        ax.set_xlabel(col_x, fontsize=11)
        ax.set_ylabel(col_y, fontsize=11)
        corr_val = df[col_x].corr(df[col_y])
        ax.set_title(
            f'Relationship: "{col_x}" vs "{col_y}"  |  Correlation: {corr_val:.2f}',
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(('Scatter Plot', fig))

    # ── CHART 5: MISSING VALUES HEATMAP ───────────────────────────────────────
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if len(missing_data) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing_pct = (missing_data / len(df) * 100).sort_values(ascending=True)
        colors = ['#D32F2F' if x > 40 else '#FF9800' if x > 10 else '#FFC107'
                  for x in missing_pct.values]
        missing_pct.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel('Missing %')
        ax.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
        ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='40% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        figures.append(('Missing Values', fig))

    return figures


# =============================================================================
# SECTION 5 — ANOMALY DETECTOR
# Finds unusual rows using IQR method across all numeric columns.
#
# HOW IT WORKS:
# For each numeric column, calculate IQR bounds.
# A row is flagged as anomalous if ANY of its numeric values is outside bounds.
# The anomaly score = how many columns have outlier values in that row.
#
# WHY ROW-LEVEL ANOMALY DETECTION:
# Column-level outlier detection tells you a VALUE is unusual.
# Row-level detection tells you a RECORD is unusual — more useful for business.
# Example: A customer with normal age but abnormally high purchase amount
#          and abnormally low account age is suspicious overall.
# =============================================================================

def detect_row_anomalies(df, top_n=10):
    """
    Detects unusual rows using a two-mode approach:

    MODE 1 — IQR outliers (classic method):
      For each numeric column, flags values outside Q1-1.5*IQR / Q3+1.5*IQR.
      Works well when data has genuine extreme values.

    MODE 2 — MAD-based ranking (for clean/uniform data):
      When IQR finds nothing (data is too uniform), ranks every row by how
      far it sits from the median using MAD (Median Absolute Deviation).
      Shows the most statistically unusual rows even without hard outliers.
      MAD is more robust than standard deviation because extreme values
      don't distort it the way they distort the mean/std.

    Binary columns (values only 0 or 1) are excluded from both methods —
    they are flag columns, not measurements, and IQR on them is meaningless.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None

    # Exclude binary columns (0/1 flag columns — IQR on them is meaningless)
    analysis_cols = [c for c in numeric_cols if df[c].nunique() > 2]
    if not analysis_cols:
        analysis_cols = numeric_cols  # fallback if everything is binary

    # ── MODE 1: IQR outlier detection ────────────────────────────────────────
    iqr_scores = pd.Series(0, index=df.index)
    for col in analysis_cols:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        is_outlier = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        iqr_scores += is_outlier.fillna(0).astype(int)

    genuine_outliers = (iqr_scores > 0).sum()

    if genuine_outliers >= 3:
        # MODE 1 — enough real outliers found, use IQR scores
        all_anomalies = iqr_scores[iqr_scores > 0].nlargest(top_n)
        if len(all_anomalies) == 0:
            return None
        result = df.loc[all_anomalies.index].copy()
        result['Anomaly Score'] = all_anomalies
        result['Detection Method'] = 'IQR outlier'
        result['Why Unusual'] = result['Anomaly Score'].apply(
            lambda x: f"Outside IQR bounds in {x} column(s)"
        )
        return result.sort_values('Anomaly Score', ascending=False)

    else:
        # MODE 2 — data is clean/uniform, use MAD-based relative ranking
        # MAD = Median Absolute Deviation — measures spread without being
        # distorted by extreme values. Each row gets a "deviation score"
        # = sum of how many MADs it sits away from median across all columns.
        mad_scores = pd.Series(0.0, index=df.index)
        for col in analysis_cols:
            clean = df[col].dropna()
            if len(clean) < 10:
                continue
            median = clean.median()
            mad = (clean - median).abs().median()
            if mad == 0:
                continue
            # Modified Z-score: 1.4826 * MAD approximates std deviation
            mod_z = (df[col] - median).abs() / (1.4826 * mad)
            mad_scores += mod_z.fillna(0)

        top_rows = mad_scores.nlargest(top_n)
        result = df.loc[top_rows.index].copy()
        result['Anomaly Score'] = top_rows.round(2)
        result['Detection Method'] = 'Relative deviation'
        result['Why Unusual'] = result['Anomaly Score'].apply(
            lambda x: f"Combined deviation score: {x:.1f} (higher = more unusual)"
        )
        return result.sort_values('Anomaly Score', ascending=False)



# =============================================================================
# SECTION 6 — NATURAL LANGUAGE QUERY ENGINE
# Answers plain-English questions about the data.
#
# HOW IT WORKS:
# Scans the question for keywords (highest, lowest, average, count, show, etc.)
# Identifies which column the question refers to.
# Runs the appropriate Pandas operation and returns the answer.
#
# WHY THIS IS POWERFUL:
# A business manager can ask "which region has highest sales"
# without knowing any Pandas syntax.
# The tool translates English to Pandas operations.
# =============================================================================

def answer_question(df, question):
    """
    Rule-based Natural Language Query Engine — v3.
    Improvements over v2:
    - Fuzzy column matching (handles typos and misspellings)
    - Multi-word column name matching (e.g. "Total Revenue")
    - 4 new query types: skewness, normality, std dev, outlier count
    - Smarter fallback with closest match suggestion
    """
    import re as re_module
    import difflib

    q = question.lower().strip().rstrip('?')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    all_cols = df.columns.tolist()

    # ── FUZZY COLUMN DETECTION ─────────────────────────────────────────────────
    # 4-pass system: exact → underscore-normalized → no-space → fuzzy token match
    def find_cols(question_text):
        q_low = question_text.lower().strip()
        found = []

        # Pass 1: exact substring match (longest first to avoid partial matches)
        for col in sorted(all_cols, key=len, reverse=True):
            if col.lower() in q_low and col not in found:
                found.append(col)

        # Pass 2: underscore-normalized — "total_revenue" matches "Total Revenue"
        q_underscore = q_low.replace(' ', '_')
        for col in all_cols:
            norm = col.lower().replace(' ', '_')
            if norm in q_underscore and col not in found:
                found.append(col)

        # Pass 3: no-space match — "totalrevenue" matches "Total Revenue"
        q_nospace = q_low.replace(' ', '')
        for col in all_cols:
            norm = col.lower().replace(' ', '')
            if norm in q_nospace and col not in found:
                found.append(col)

        # Pass 4: fuzzy match — handles typos like "salry" → "salary"
        if not found:
            q_tokens = q_low.replace('?', '').split()
            for col in all_cols:
                col_tokens = col.lower().split()
                match_score = all(
                    any(difflib.SequenceMatcher(None, ct, qt).ratio() > 0.78
                        for qt in q_tokens)
                    for ct in col_tokens
                )
                if match_score and col not in found:
                    found.append(col)

        return found

    mentioned_cols = find_cols(q)
    mentioned_col = mentioned_cols[0] if mentioned_cols else None
    second_col = mentioned_cols[1] if len(mentioned_cols) > 1 else None

    # ── KEYWORD NORMALIZATION — handle common typos and variants ───────────────
    # Map misspellings to canonical keywords
    typo_map = {
        'averge': 'average', 'avrage': 'average', 'averg': 'average',
        'minium': 'minimum', 'minmum': 'minimum',
        'maximun': 'maximum', 'maxmum': 'maximum',
        'corelation': 'correlation', 'coorelation': 'correlation',
        'standrd': 'standard', 'stadard': 'standard',
        'skewnes': 'skewness', 'skewnes': 'skewness',
        'distribtion': 'distribution', 'distibution': 'distribution',
        'normaly': 'normally', 'normall': 'normally',
    }
    for typo, fix in typo_map.items():
        q = q.replace(typo, fix)

    # ── EXTRACT NUMBER FROM QUESTION ───────────────────────────────────────────
    numbers = re_module.findall(r'\b\d+\b', q)
    n = int(numbers[0]) if numbers else 5
    n = min(max(n, 1), 50)

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    # Dataset size
    if any(k in q for k in ['how many rows', 'total rows', 'row count', 'number of rows']):
        return f"The dataset has **{len(df):,} rows**."

    if any(k in q for k in ['dataset size', 'how big', 'how large', 'size of']):
        return f"The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**."

    # Column info
    if any(k in q for k in ['how many columns', 'total columns', 'column count',
                              'what columns', 'column names', 'list columns',
                              'show columns', 'what are the columns']):
        numeric_list = ', '.join(numeric_cols) if numeric_cols else 'None'
        text_list = ', '.join(text_cols) if text_cols else 'None'
        return (f"**{len(df.columns)} columns:**\n"
                f"Numeric: {numeric_list}\n"
                f"Text: {text_list}")

    # Which [text] has highest/lowest [numeric]? — groupby aggregation
    if any(k in q for k in ['which', 'what']):
        if any(k in q for k in ['highest', 'maximum', 'max', 'largest', 'most', 'best', 'top']):
            if mentioned_col and second_col:
                group_col = mentioned_col if mentioned_col in text_cols else second_col
                val_col = second_col if mentioned_col in text_cols else mentioned_col
                if val_col in numeric_cols and group_col in text_cols:
                    result = df.groupby(group_col)[val_col].mean().idxmax()
                    val = df.groupby(group_col)[val_col].mean().max()
                    return f"**{group_col} with highest average {val_col}:** {result} ({val:,.2f})"
            elif mentioned_col and mentioned_col in numeric_cols:
                max_row = df.loc[df[mentioned_col].idxmax()]
                details = ' | '.join([f"{c}: {max_row[c]}" for c in df.columns[:4]])
                return f"**Highest {mentioned_col}:** {df[mentioned_col].max():,.2f}\n{details}"
        if any(k in q for k in ['lowest', 'minimum', 'min', 'smallest', 'least', 'worst', 'bottom']):
            if mentioned_col and second_col:
                group_col = mentioned_col if mentioned_col in text_cols else second_col
                val_col = second_col if mentioned_col in text_cols else mentioned_col
                if val_col in numeric_cols and group_col in text_cols:
                    result = df.groupby(group_col)[val_col].mean().idxmin()
                    val = df.groupby(group_col)[val_col].mean().min()
                    return f"**{group_col} with lowest average {val_col}:** {result} ({val:,.2f})"

    # Highest / maximum
    if any(k in q for k in ['highest', 'maximum', 'max', 'largest', 'most']):
        if mentioned_col and mentioned_col in numeric_cols:
            max_row = df.loc[df[mentioned_col].idxmax()]
            details = ' | '.join([f"{c}: {max_row[c]}" for c in df.columns[:4]])
            return f"**Highest {mentioned_col}:** {df[mentioned_col].max():,.2f}\n{details}"
        elif mentioned_col and mentioned_col in text_cols:
            top = df[mentioned_col].value_counts().head(5)
            return f"**Most frequent in {mentioned_col}:**\n" + "\n".join([f"  {v}: {c:,}" for v, c in top.items()])

    # Lowest / minimum
    if any(k in q for k in ['lowest', 'minimum', 'min', 'smallest', 'least']):
        if mentioned_col and mentioned_col in numeric_cols:
            min_row = df.loc[df[mentioned_col].idxmin()]
            details = ' | '.join([f"{c}: {min_row[c]}" for c in df.columns[:4]])
            return f"**Lowest {mentioned_col}:** {df[mentioned_col].min():,.2f}\n{details}"

    # Average / mean
    if any(k in q for k in ['average', 'mean', 'avg']):
        if mentioned_col and mentioned_col in numeric_cols:
            return (f"**{mentioned_col}** — "
                    f"Average: {df[mentioned_col].mean():,.2f} | "
                    f"Median: {df[mentioned_col].median():,.2f}")
        elif numeric_cols:
            results = [f"  {col}: {df[col].mean():,.2f}" for col in numeric_cols[:6]]
            return "**Averages across numeric columns:**\n" + "\n".join(results)

    # Sum / total
    if any(k in q for k in ['sum', 'total']):
        if mentioned_col and mentioned_col in numeric_cols:
            return f"**Total {mentioned_col}:** {df[mentioned_col].sum():,.2f}"
        elif numeric_cols:
            results = [f"  {col}: {df[col].sum():,.2f}" for col in numeric_cols[:6]]
            return "**Totals across numeric columns:**\n" + "\n".join(results)

    # Standard deviation
    if any(k in q for k in ['standard deviation', 'std dev', 'std', 'variance', 'variability']):
        if mentioned_col and mentioned_col in numeric_cols:
            return (f"**{mentioned_col} variability:**\n"
                    f"  Std Dev: {df[mentioned_col].std():,.2f}\n"
                    f"  Variance: {df[mentioned_col].var():,.2f}\n"
                    f"  Range: {df[mentioned_col].min():,.2f} to {df[mentioned_col].max():,.2f}")
        elif numeric_cols:
            results = [f"  {col}: {df[col].std():,.2f}" for col in numeric_cols[:6]]
            return "**Standard deviations:**\n" + "\n".join(results)

    # Skewness
    if any(k in q for k in ['skew', 'skewness']):
        if mentioned_col and mentioned_col in numeric_cols:
            skew = df[mentioned_col].skew()
            if skew > 0.5:
                interp = "right-skewed — most values are low, a few very high values pull the average up"
            elif skew < -0.5:
                interp = "left-skewed — most values are high, a few very low values pull the average down"
            else:
                interp = "approximately symmetric — values are evenly distributed"
            return f"**Skewness of {mentioned_col}:** {skew:.3f}\n{interp}"
        elif numeric_cols:
            results = []
            for col in numeric_cols[:5]:
                skew = df[col].skew()
                tag = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "symmetric"
                results.append(f"  {col}: {skew:.3f} ({tag})")
            return "**Skewness by column:**\n" + "\n".join(results)

    # Normality
    if any(k in q for k in ['normal', 'normally distributed', 'normality', 'gaussian']):
        if mentioned_col and mentioned_col in numeric_cols:
            from scipy import stats as scipy_stats
            if len(df) >= 3:
                stat, p = scipy_stats.shapiro(df[mentioned_col].dropna())
                verdict = "likely normal (p > 0.05)" if p > 0.05 else "not normal (p ≤ 0.05)"
                return (f"**Normality test for {mentioned_col} (Shapiro-Wilk):**\n"
                        f"  p-value: {p:.4f} → {verdict}\n"
                        f"  Skewness: {df[mentioned_col].skew():.3f}")
        return "Specify a numeric column. Example: 'is salary normally distributed?'"

    # Outlier count
    if any(k in q for k in ['outlier', 'outliers', 'anomal']):
        if mentioned_col and mentioned_col in numeric_cols:
            q1 = df[mentioned_col].quantile(0.25)
            q3 = df[mentioned_col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[mentioned_col] < q1 - 1.5*iqr) | (df[mentioned_col] > q3 + 1.5*iqr)]
            return (f"**Outliers in {mentioned_col} (IQR method):**\n"
                    f"  Count: {len(outliers):,} of {len(df):,} rows ({len(outliers)/len(df)*100:.1f}%)\n"
                    f"  Normal range: {q1 - 1.5*iqr:,.2f} to {q3 + 1.5*iqr:,.2f}")
        elif numeric_cols:
            results = []
            for col in numeric_cols[:5]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                count = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
                results.append(f"  {col}: {count} outliers")
            return "**Outlier counts (IQR method):**\n" + "\n".join(results)

    # Count / unique values
    if any(k in q for k in ['count', 'unique', 'distinct']):
        if mentioned_col:
            unique_count = df[mentioned_col].nunique()
            top5 = ', '.join(str(v) for v in df[mentioned_col].value_counts().head(5).index)
            return f"**{mentioned_col}** — {unique_count:,} unique values\nTop 5: {top5}"
        else:
            results = [f"  {col}: {df[col].nunique():,} unique" for col in df.columns[:8]]
            return "**Unique value counts:**\n" + "\n".join(results)

    # Show top N
    if any(k in q for k in ['top', 'show', 'display', 'first']):
        if mentioned_col and mentioned_col in numeric_cols:
            top_rows = df.nlargest(n, mentioned_col).head(n)
            return f"**Top {n} rows by {mentioned_col}:**\n{top_rows.to_string(index=False)}"
        elif mentioned_col and mentioned_col in text_cols:
            top_vals = df[mentioned_col].value_counts().head(n)
            return f"**Top {n} values in {mentioned_col}:**\n" + "\n".join([f"  {v}: {c:,}" for v, c in top_vals.items()])
        else:
            return f"**First {n} rows:**\n{df.head(n).to_string(index=False)}"

    # Missing values
    if any(k in q for k in ['missing', 'null', 'empty', 'nan', 'incomplete', 'na ']):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            return "**No missing values found in this dataset.** ✅"
        result = "**Missing values by column:**\n"
        for col, count in missing.items():
            pct = count / len(df) * 100
            result += f"  {col}: {count:,} ({pct:.1f}%)\n"
        return result

    # Distribution / spread
    if any(k in q for k in ['distribution', 'spread', 'range']):
        if mentioned_col and mentioned_col in numeric_cols:
            col_data = df[mentioned_col]
            return (f"**Distribution of {mentioned_col}:**\n"
                    f"  Min: {col_data.min():,.2f}  |  Max: {col_data.max():,.2f}\n"
                    f"  Mean: {col_data.mean():,.2f}  |  Median: {col_data.median():,.2f}\n"
                    f"  Std Dev: {col_data.std():,.2f}  |  Skewness: {col_data.skew():.3f}")

    # Correlation
    if any(k in q for k in ['correlation', 'correlated', 'relationship', 'related']):
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            strong = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        direction = "positive" if val > 0 else "negative"
                        strong.append(f"  {numeric_cols[i]} ↔ {numeric_cols[j]}: {val:.2f} ({direction})")
            if strong:
                return "**Strong correlations (|r| > 0.5):**\n" + "\n".join(strong)
            else:
                return "No strong correlations found (all |r| < 0.5)."
        return "Need at least 2 numeric columns to calculate correlation."

    # Describe / summary — formatted cleanly
    if any(k in q for k in ['describe', 'summary', 'statistics', 'stats', 'overview']):
        if mentioned_col and mentioned_col in numeric_cols:
            desc = df[mentioned_col].describe()
            return (
                f"**Statistics for {mentioned_col}:**\n"
                f"  Count:  {desc['count']:,.0f}\n"
                f"  Mean:   {desc['mean']:,.2f}\n"
                f"  Median: {df[mentioned_col].median():,.2f}\n"
                f"  Std:    {desc['std']:,.2f}\n"
                f"  Min:    {desc['min']:,.2f}\n"
                f"  25%:    {desc['25%']:,.2f}\n"
                f"  75%:    {desc['75%']:,.2f}\n"
                f"  Max:    {desc['max']:,.2f}"
            )
        # Full dataset describe — formatted as readable lines
        lines_out = [f"**Dataset Summary — {len(df):,} rows × {len(df.columns)} columns**\n"]
        if numeric_cols:
            lines_out.append("**Numeric columns:**")
            for col in numeric_cols:
                col_data = df[col].dropna()
                lines_out.append(
                    f"  {col}: mean={col_data.mean():,.2f}, "
                    f"min={col_data.min():,.2f}, "
                    f"max={col_data.max():,.2f}, "
                    f"nulls={df[col].isnull().sum()}"
                )
        if text_cols:
            lines_out.append("\n**Text columns:**")
            for col in text_cols:
                unique = df[col].nunique()
                top = df[col].value_counts().index[0] if len(df[col].dropna()) > 0 else 'N/A'
                lines_out.append(f"  {col}: {unique} unique values, most common: {top}")
        return "\n".join(lines_out)

    # Compare two columns
    if any(k in q for k in ['compare', 'versus', 'vs']):
        if mentioned_col and second_col and mentioned_col in numeric_cols and second_col in numeric_cols:
            return (f"**Comparison: {mentioned_col} vs {second_col}**\n"
                    f"  {mentioned_col} — Mean: {df[mentioned_col].mean():,.2f}, Max: {df[mentioned_col].max():,.2f}\n"
                    f"  {second_col} — Mean: {df[second_col].mean():,.2f}, Max: {df[second_col].max():,.2f}")

    # ── SMART FALLBACK ─────────────────────────────────────────────────────────
    # Try to find closest column match and suggest the right question
    if mentioned_col is None:
        # Suggest closest column name using fuzzy matching
        q_words = q.split()
        best_match = None
        best_score = 0
        for col in all_cols:
            for word in q_words:
                score = difflib.SequenceMatcher(None, word, col.lower()).ratio()
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = col

        if best_match:
            return (f"Did you mean **{best_match}**? Try:\n"
                    f"- 'what is average {best_match}'\n"
                    f"- 'show top 10 {best_match}'\n"
                    f"- 'highest {best_match}'")

        return (f"I could not match your question to a column.\n"
                f"Available columns: **{', '.join(df.columns.tolist())}**\n\n"
                f"Try: 'what is average salary' | 'show top 10 city' | 'which department has highest salary'")

    return (f"I understand you are asking about **{mentioned_col}**. "
            "Try being more specific:\n"
            f"- 'what is average {mentioned_col}'\n"
            f"- 'highest {mentioned_col}'\n"
            f"- 'which [category] has highest {mentioned_col}'\n"
            f"- 'is {mentioned_col} normally distributed'\n"
            f"- 'skewness of {mentioned_col}'")


# =============================================================================
# SECTION 7 — SAMPLE DATASETS
# =============================================================================

def get_sample_data(choice):
    """Returns sample datasets for demonstration."""
    np.random.seed(42)

    if choice == "Retail Sales":
        dates = pd.date_range('2022-01-01', periods=200, freq='D')
        return pd.DataFrame({
            'date': dates,
            'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], 200),
            'sales': np.random.normal(50000, 15000, 200).clip(5000),
            'units': np.random.randint(10, 500, 200),
            'customers': np.random.randint(20, 300, 200),
            'discount_pct': np.random.uniform(0, 30, 200).round(1)
        })

    elif choice == "Student Performance":
        # Realistic: students who study more get better scores across all exams
        np.random.seed(42)
        n = 150
        base_ability = np.random.normal(70, 12, n)          # underlying ability
        return pd.DataFrame({
            'student_id':       range(1, n+1),
            'gender':           np.random.choice(['Male', 'Female'], n),
            'department':       np.random.choice(['CS', 'IT', 'ECE', 'Mechanical', 'Civil'], n),
            'attendance_pct':   (base_ability * 0.9 + np.random.normal(0, 8, n)).clip(40, 100).round(1),
            'assignment_score': (base_ability + np.random.normal(0, 10, n)).clip(0, 100).round(1),
            'midterm_score':    (base_ability * 0.95 + np.random.normal(0, 10, n)).clip(0, 100).round(1),
            'final_score':      (base_ability + np.random.normal(0, 8, n)).clip(0, 100).round(1),
            'cgpa':             ((base_ability / 100) * 6 + np.random.normal(0, 0.6, n)).clip(4, 10).round(2)
        })

    elif choice == "Employee Data":
        # Realistic: salary increases with experience, performance affects projects
        np.random.seed(42)
        n = 100
        experience = np.random.randint(0, 20, n)
        performance = np.random.normal(3.5, 0.8, n).clip(1, 5).round(1)
        return pd.DataFrame({
            'employee_id':        range(1, n+1),
            'department':         np.random.choice(['Engineering','Sales','HR','Finance','Marketing'], n),
            'experience_years':   experience,
            'salary':             (30000 + experience * 2500 + np.random.normal(0, 8000, n)).clip(25000).round(-3),
            'performance_score':  performance,
            'projects_completed': (performance * 4 + np.random.normal(0, 2, n)).clip(1, 25).round().astype(int),
            'city':               np.random.choice(['Mumbai','Pune','Bangalore','Delhi','Chennai'], n),
            'left_company':       np.random.choice(['Yes','No'], n, p=[0.2, 0.8])
        })


# =============================================================================
# SECTION 8 — MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Smart Auto-Analytics",
        page_icon="🔬",
        layout="wide"
    )

    # ── TAB SIZE CSS + GLOBAL DARK-MODE SAFE STYLES ──────────────────────────
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: #1e1e2e;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1565C0 !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.title("🔬 Smart Auto-Analytics and Insights Generator")
    st.markdown(
        "*Upload any dataset — sales, students, employees, anything. "
        "Get automatic profiling, smart insights, visualizations, "
        "anomaly detection, and natural language queries — instantly.*  \n"
        "Built by **Kartikeya Warhade**"
    )
    st.markdown("---")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Data Source")
    source = st.sidebar.radio(
        "Choose data source",
        ["📁 Upload Your File", "📊 Use Sample Dataset"]
    )

    df = None

    if source == "📊 Use Sample Dataset":
        sample_choice = st.sidebar.selectbox(
            "Select Sample Dataset",
            ["Retail Sales", "Student Performance", "Employee Data"]
        )
        df = get_sample_data(sample_choice)
        st.sidebar.success(f"Loaded: {sample_choice} ({len(df):,} rows)")

    else:
        uploaded = st.sidebar.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"]
        )

        if uploaded is None:
            # Landing page
            st.info("👈 Upload your file from the sidebar, or try a Sample Dataset.")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 📋 Data Profiling")
                st.markdown("Complete column-by-column analysis of types, missing values, and distributions.")
            with c2:
                st.markdown("### 💡 Smart Insights")
                st.markdown("Automatically finds the 5 most interesting patterns in your data.")
            with c3:
                st.markdown("### 🔍 Anomaly Detection")
                st.markdown("Flags unusual rows using IQR method across all numeric columns.")

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown("### 📊 Auto Charts")
                st.markdown("Right chart type selected automatically for each column.")
            with c5:
                st.markdown("### 💬 Ask Your Data")
                st.markdown("Type plain-English questions. Get instant answers.")
            with c6:
                st.markdown("### 📥 Download")
                st.markdown("Export profile report and cleaned data as CSV.")

            st.markdown("---")
            st.markdown("**Accepted:** Any CSV or Excel file with headers in the first row.")
            return

        df, err = load_file(uploaded)
        if err:
            st.error(f"❌ {err}")
            return

        if len(df) < 5:
            st.error("❌ File has fewer than 5 rows. Please upload a larger dataset.")
            return

        st.sidebar.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    if df is None:
        return

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Profile",
        "💡 Smart Insights",
        "📊 Auto Charts",
        "🔍 Anomaly Detection",
        "💬 Ask Your Data"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA PROFILE
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Data Profile")
        st.markdown("Complete breakdown of every column in your dataset.")

        # Quick stats row — HTML cards (dark-mode safe, no white boxes)
        numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols     = df.select_dtypes(include=['object']).columns.tolist()
        total_missing = df.isnull().sum().sum()
        duplicates    = int(df.duplicated().sum())

        def stat_card(value, label, bg="#1565C0"):
            return (
                f'<div style="flex:1;min-width:130px;background:{bg};border-radius:10px;'
                f'padding:16px 12px;text-align:center;color:#fff;'
                f'box-shadow:0 2px 6px rgba(0,0,0,0.3);margin:4px;">'
                f'<div style="font-size:24px;font-weight:800;">{value}</div>'
                f'<div style="font-size:10px;margin-top:5px;opacity:0.85;'
                f'text-transform:uppercase;letter-spacing:0.8px;">{label}</div>'
                f'</div>'
            )

        st.markdown(
            '<div style="display:flex;gap:10px;margin:12px 0 20px 0;flex-wrap:wrap;">'
            + stat_card(f"{len(df):,}",      "Total Rows",      "#1565C0")
            + stat_card(str(len(df.columns)),"Total Columns",   "#1565C0")
            + stat_card(str(len(numeric_cols)),"Numeric Cols",  "#37474F")
            + stat_card(f"{total_missing:,}", "Missing Values", "#B71C1C" if total_missing > 0 else "#2E7D32")
            + stat_card(str(duplicates),       "Duplicate Rows", "#B71C1C" if duplicates > 0 else "#2E7D32")
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Full profile table
        st.subheader("Column-by-Column Profile")
        profile_df = profile_data(df)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        # Statistical summary for numeric columns
        if numeric_cols:
            st.subheader("Statistical Summary — Numeric Columns")
            st.dataframe(
                df[numeric_cols].describe().round(2),
                use_container_width=True
            )

        # Downloads — profile + full dataset
        st.subheader("📥 Downloads")
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_buf = io.StringIO()
            profile_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download Profile Report (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            full_buf = io.StringIO()
            df.to_csv(full_buf, index=False)
            st.download_button(
                label="⬇️ Download Full Dataset (CSV)",
                data=full_buf.getvalue(),
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — SMART INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("Smart Insights")
        st.markdown(
            "Automatically generated observations about your data. "
            "Each insight is backed by a specific calculation."
        )

        with st.spinner("Analyzing data for insights..."):
            insights = generate_insights(df)

        for insight in insights:
            icon = insight['icon']
            title = insight['title']
            detail = insight['detail']
            itype = insight['type']

            if itype == 'warning':
                st.warning(f"**{icon} {title}**\n\n{detail}")
            elif itype == 'info':
                st.info(f"**{icon} {title}**\n\n{detail}")
            else:
                st.success(f"**{icon} {title}**\n\n{detail}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — AUTO CHARTS
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("Auto-Generated Visualizations")
        st.markdown(
            "Charts automatically selected based on column types. "
            "Histograms for distributions, bar charts for categories, "
            "heatmap for correlations."
        )

        with st.spinner("Generating visualizations..."):
            figures = create_auto_visualizations(df)

        if figures:
            for title, fig in figures:
                st.subheader(title)
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("---")
        else:
            st.info("No suitable columns found for visualization.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — ANOMALY DETECTION
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("Anomaly Detection")
        st.markdown("""
        **Method: IQR-based row-level anomaly detection**

        For each numeric column, the IQR (Interquartile Range) bounds are calculated.
        A row gets an anomaly score equal to the number of columns where its value falls outside the IQR bounds.
        Higher score = more unusual the row is.
        """)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found for anomaly detection.")
        else:
            show_all = st.checkbox("Show all results", value=False)
            if show_all:
                top_n = len(df)
                st.caption(f"Showing all rows (up to {len(df)} total)")
            else:
                top_n = st.slider("Show top N most anomalous rows", 1, 30, 10)

            with st.spinner("Detecting anomalies..."):
                anomaly_df = detect_row_anomalies(df, top_n)

            if anomaly_df is not None and len(anomaly_df) > 0:
                method = anomaly_df['Detection Method'].iloc[0]

                if method == 'IQR outlier':
                    # Mode 1: genuine outliers found
                    total_anomaly_df = detect_row_anomalies(df, top_n=len(df))
                    total_anomalies = len(total_anomaly_df) if total_anomaly_df is not None else 0
                    if len(anomaly_df) < total_anomalies:
                        st.warning(f"⚠️ {total_anomalies} rows fall outside IQR bounds. Showing top {len(anomaly_df)} by anomaly score.")
                    else:
                        st.warning(f"⚠️ Found {total_anomalies} rows with values outside IQR bounds.")
                else:
                    # Mode 2: no hard outliers — showing relative ranking
                    st.info(
                        f"📊 This dataset has no extreme outliers — all values are within normal statistical range. "
                        f"Showing the **top {len(anomaly_df)} most unusual rows** by relative deviation from the median. "
                        f"Higher score = further from typical values across multiple columns."
                    )

                st.dataframe(anomaly_df, use_container_width=True)

                # Score distribution chart (only meaningful in IQR mode)
                if method == 'IQR outlier':
                    analysis_cols = [c for c in numeric_cols if df[c].nunique() > 2]
                    all_scores = pd.Series(0, index=df.index)
                    for col in analysis_cols:
                        clean = df[col].dropna()
                        if len(clean) < 10:
                            continue
                        Q1, Q3 = clean.quantile(0.25), clean.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR == 0:
                            continue
                        is_out = (df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)
                        all_scores += is_out.fillna(0).astype(int)

                    fig_a, ax_a = plt.subplots(figsize=(8, 4))
                    score_counts = all_scores.value_counts().sort_index()
                    ax_a.bar(score_counts.index.astype(str), score_counts.values,
                            color=['#4CAF50' if i == 0 else '#FF9800' if i == 1 else '#D32F2F'
                                   for i in score_counts.index])
                    ax_a.set_xlabel("Anomaly Score (number of columns where row is an outlier)")
                    ax_a.set_ylabel("Number of Rows")
                    ax_a.set_title("Distribution of Anomaly Scores")
                    ax_a.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_a)
                    plt.close()
            else:
                st.success("✅ No anomalies detected in this dataset.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — ASK YOUR DATA
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.header("Ask Your Data")
        st.markdown(
            "Type a plain-English question about your dataset. "
            "No code needed."
        )

        # Dynamic example questions based on dataset columns
        st.markdown("**Example questions you can ask:**")

        numeric_cols_q = df.select_dtypes(include=['number']).columns.tolist()
        text_cols_q = df.select_dtypes(include=['object']).columns.tolist()

        # Base questions — always work on any dataset
        examples = [
            "How many rows?",
            "What columns are there?",
            "How many missing values?",
        ]

        # Correlation only shown if dataset has 2+ numeric columns
        if len(numeric_cols_q) >= 2:
            examples.append("What is the correlation?")

        # Numeric-specific questions using actual column names
        if numeric_cols_q:
            examples.append(f"What is average {numeric_cols_q[0]}?")
            examples.append(f"What is minimum {numeric_cols_q[0]}?")
            if len(numeric_cols_q) > 1:
                examples.append(f"What is maximum {numeric_cols_q[1]}?")

        # Text-specific questions using actual column names
        if text_cols_q:
            examples.append(f"Show top 10 {text_cols_q[0]}")
            # Only show groupby question if both a text col and numeric col exist
            if text_cols_q and numeric_cols_q:
                examples.append(f"Which {text_cols_q[0]} has highest {numeric_cols_q[0]}?")

        # Summary question — always works
        examples.append("Describe the dataset")

        cols_ex = st.columns(4)
        for i, ex in enumerate(examples):
            if cols_ex[i % 4].button(ex, key=f"ex_{i}"):
                st.session_state['question'] = ex

        st.markdown("---")

        question = st.text_input(
            "Your question:",
            value=st.session_state.get('question', ''),
            placeholder="e.g. what is average sales  |  show top 5 region  |  how many missing values"
        )

        if question:
            with st.spinner("Finding answer..."):
                answer = answer_question(df, question)
            st.markdown("### Answer:")
            st.markdown(answer)

        # Show data columns reminder
        with st.expander("📋 Available columns in your dataset — click to see column names before asking questions"):
            col_info = []
            for col in df.columns:
                dtype = "Numeric" if col in df.select_dtypes(include=[np.number]).columns else "Text"
                col_info.append(f"**{col}** ({dtype})")
            st.markdown("  |  ".join(col_info))


if __name__ == "__main__":
    main()
