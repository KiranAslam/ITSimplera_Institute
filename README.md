# Online Retail II — Exploratory Data Analysis

A focused exploratory data analysis on the UCI "Online Retail II" dataset (2009–2011) to understand customer purchase behaviour, top-selling products, seasonality and country-level revenue — preparing the data and insights for a future recommendation system.

## Files
- Week_01/Online_Retail_II_EDA.ipynb — Jupyter notebook with the full EDA, visualizations and computed metrics.
- Week_01/Data/online_retail_II.xlsx — Original dataset (two sheets: 2009-2010, 2010-2011).

## Key Insights (from the notebook)
- Top revenue product: REGENCY CAKESTAND 3 TIER (£327,813.65).
- Top volume product: WORLD WAR 2 GLIDERS ASSTD DESIGNS (108,545 units).
- Peak revenue month: 2011-11 (holiday season spike).
- Geographic concentration: United Kingdom accounts for ~84.94% of total revenue.
- Data gap: ~22.77% of transactions missing `CustomerID` (limits personalization).
- Outliers: Quantity outliers ≈ 10.91% and Price outliers ≈ 6.38% — often wholesale/cancellations.
- Correlations: Quantity ↔ Revenue = 0.76 (strong), Price ↔ Revenue = 0.06 (weak).

## How to run
1. Clone or download the repo.
2. Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install pandas numpy matplotlib seaborn openpyxl
```

3. Open and run the notebook:
- Launch Jupyter or VS Code and open `Week_01/Online_Retail_II_EDA.ipynb`.
- Execute cells sequentially to reproduce the analysis and figures.

## Notes and next steps
- Fix missing `CustomerID` records or investigate guest checkout flows before building a recommender.
- Treat outliers carefully: many are legitimate B2B wholesale orders and cancellations.
- Consider adding a `requirements.txt` or environment file for reproducibility.

## License & Contact
This repository is provided for educational and analysis purposes. For questions, contact the author: Kiran Aslam.
