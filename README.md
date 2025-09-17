# PE-Port-Analytics
July 2025 | Private Equity Portfolio Analytics using IRR and TVPI


# ===============================
# app.py - Full Version with Hardcoded Data
# ===============================
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import streamlit as st

# ===============================
# 0. Hardcoded Test Data
# ===============================
fundamentals = pd.DataFrame([
    {'ticker': 'AAPL', 'date': '2023-03-31', 'revenue': 1200, 'sector': 'Tech'},
    {'ticker': 'AAPL', 'date': '2023-06-30', 'revenue': 1300, 'sector': 'Tech'},
    {'ticker': 'AAPL', 'date': '2023-09-30', 'revenue': 1250, 'sector': 'Tech'},
    {'ticker': 'MSFT', 'date': '2023-03-31', 'revenue': 1000, 'sector': 'Tech'},
    {'ticker': 'MSFT', 'date': '2023-06-30', 'revenue': 1100, 'sector': 'Tech'},
    {'ticker': 'MSFT', 'date': '2023-09-30', 'revenue': 1050, 'sector': 'Tech'},
    {'ticker': 'GOOG', 'date': '2023-03-31', 'revenue': 900, 'sector': 'Tech'},
    {'ticker': 'GOOG', 'date': '2023-06-30', 'revenue': 950, 'sector': 'Tech'},
    {'ticker': 'GOOG', 'date': '2023-09-30', 'revenue': 970, 'sector': 'Tech'},
])

returns = pd.DataFrame([
    {'ticker': 'AAPL', 'date': '2023-03-31', 'quarterly_return': 0.05},
    {'ticker': 'AAPL', 'date': '2023-06-30', 'quarterly_return': 0.02},
    {'ticker': 'AAPL', 'date': '2023-09-30', 'quarterly_return': -0.01},
    {'ticker': 'MSFT', 'date': '2023-03-31', 'quarterly_return': 0.03},
    {'ticker': 'MSFT', 'date': '2023-06-30', 'quarterly_return': 0.04},
    {'ticker': 'MSFT', 'date': '2023-09-30', 'quarterly_return': -0.02},
    {'ticker': 'GOOG', 'date': '2023-03-31', 'quarterly_return': 0.06},
    {'ticker': 'GOOG', 'date': '2023-06-30', 'quarterly_return': 0.01},
    {'ticker': 'GOOG', 'date': '2023-09-30', 'quarterly_return': 0.00},
])

# ===============================
# 1. Clean Data
# ===============================
fundamentals['ticker'] = fundamentals['ticker'].str.upper().str.strip()
fundamentals['revenue'] = fundamentals['revenue'].apply(lambda x: np.nan if x < 0 else x)
fundamentals['revenue'] = fundamentals['revenue'].fillna(fundamentals['revenue'].median())
fundamentals['date'] = pd.to_datetime(fundamentals['date'])
returns['date'] = pd.to_datetime(returns['date'])

# ===============================
# 2. Grouping and Pivoting
# ===============================
quarterly_rev = (
    fundamentals.groupby(['ticker', pd.Grouper(key='date', freq='Q')])['revenue']
    .sum()
    .reset_index()
)
rev_pivot = quarterly_rev.pivot(index='date', columns='ticker', values='revenue')

# ===============================
# 3. Merge with Returns
# ===============================
merged = quarterly_rev.merge(returns, on=['ticker', 'date'], how='inner')

# ===============================
# 4. EDA
# ===============================
merged_summary = merged.describe(include='all')
missing = merged.isna().mean().sort_values(ascending=False)

# Correlation
corr = merged[['revenue', 'quarterly_return']].corr()

# ===============================
# 5. Backtesting Signal
# ===============================
merged['rev_growth'] = merged.groupby('ticker')['revenue'].pct_change()
merged['strategy_return'] = np.sign(merged['rev_growth']).fillna(0) * merged['quarterly_return']
merged['cum_strategy'] = (1 + merged['strategy_return']).cumprod()
merged['cum_market'] = (1 + merged['quarterly_return']).cumprod()

# ===============================
# 6. Regression
# ===============================
reg_data = merged.dropna(subset=['rev_growth', 'quarterly_return'])
X = reg_data[['rev_growth']].values
y = reg_data['quarterly_return'].values
model = LinearRegression()
model.fit(X, y)


correlation =df[“y”].corr(df(“x”]
Print(correlation) 

#Lag analysis
from statsmodel.tsa.stattools import CCs

cross_corr = ccf(df[“Advertising_Spend”], df[“Sales_Revenue”]
print(cross_corr)

# ===============================
# 7. Streamlit Dashboard
# ===============================
st.title("Revenue Growth Backtest Dashboard")

st.subheader("Pivoted Quarterly Revenue")
st.dataframe(rev_pivot)

st.subheader("Merged Data Summary")
st.write(merged_summary)

st.subheader("Missingness by Column")
st.write(missing)

st.subheader("Correlation Heatmap")
fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(fig_corr)

st.subheader("Distribution of Quarterly Returns")
fig_hist = px.histogram(merged, x="quarterly_return", nbins=10, title="Distribution of Quarterly Returns")
st.plotly_chart(fig_hist)

st.subheader("Sector-level Revenue")
sector_rev = fundamentals.groupby("sector")['revenue'].sum().reset_index()
fig_sector = px.bar(sector_rev, x="sector", y="revenue", title="Revenue by Sector")
st.plotly_chart(fig_sector)

st.subheader("Time Series per Ticker")
selected_ticker = st.selectbox("Select Ticker", merged['ticker'].unique())
ticker_data = merged[merged['ticker'] == selected_ticker]
fig_line = px.line(ticker_data, x="date", y="revenue", title=f"Revenue Over Time for {selected_ticker}")
st.plotly_chart(fig_line)

st.subheader("Backtest: Strategy vs Market")
fig_backtest = px.line(ticker_data, x="date", y=["cum_strategy", "cum_market"], title=f"Backtest for {selected_ticker}")
st.plotly_chart(fig_backtest)

st.subheader("Regression: Revenue Growth vs Quarterly Returns")
st.write(f"Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}, R²: {model.score(X, y):.4f}")
fig_reg = px.scatter(reg_data, x="rev_growth", y="quarterly_return", trendline="ols",
                     title="Regression: Revenue Growth vs Quarterly Returns")
st.plotly_chart(fig_reg)
