# Online Retail II — Exploratory Data Analysis

This repository contains an end-to-end exploratory data analysis of the UCI "Online Retail II" dataset (2009–2011). The notebooks explore purchase behaviour, top-selling products, country-level revenue, seasonality, outliers and correlation structure — with the goal of preparing clean, well-understood inputs for a future recommendation system.

## Files
- `Week_01/Online_Retail_II_EDA.ipynb` — Jupyter notebook with the full EDA, visualizations and computed metrics.
- `Week_01/Data/online_retail_II.xlsx` — Original dataset (two sheets: 2009-2010, 2010-2011).
- `Week_01/figures/` — (recommended) folder to store exported plot images used in this README.

## Key Insights (quick)
- Top revenue product: REGENCY CAKESTAND 3 TIER (£327,813.65)
- Top volume product: WORLD WAR 2 GLIDERS ASSTD DESIGNS (108,545 units)
- Peak revenue month: 2011-11 (holiday season spike)
- Geographic concentration: United Kingdom accounts for ~84.94% of total revenue
- Data gap: ~22.77% of transactions missing `CustomerID` (limits personalization)

## Visualizations and explanations
Below are the recommended visualizations produced in the notebook. To include the images in this README, run the plotting cell in the notebook and save the figures to `Week_01/figures/` using the example snippets.

- **Top 10 Products — by Quantity**
  - File: `Week_01/figures/top10_qty.png`
  - Why: Shows which items sell most units (volume leaders). Useful for inventory and promotion planning.
  - Insight: High-unit items are not always high-revenue items; volume leaders may have low margins.

- **Top 10 Products — by Revenue**
  - File: `Week_01/figures/top10_rev.png`
  - Why: Identifies the products that contribute most to revenue — critical for prioritizing merchandising and cross-sell strategies.
  - Insight: `REGENCY CAKESTAND 3 TIER` drives the most revenue but is not the top volume seller.

- **Top Countries by Revenue**
  - File: `Week_01/figures/top_countries.png`
  - Why: Reveals geographic concentration risks and opportunities for international expansion.
  - Insight: United Kingdom contributes ~85% of revenue — heavy domestic dependence.

- **Monthly Revenue Trend**
  - File: `Week_01/figures/monthly_revenue.png`
  - Why: Shows seasonality and peak demand periods (critical for planning inventory and marketing).
  - Insight: Clear spikes in November (holiday season).

- **Correlation Heatmap**
  - File: `Week_01/figures/corr_heatmap.png`
  - Why: Quantifies relationships between `Quantity`, `Price`, `Revenue` and (if present) `CustomerID`.
  - Insight: Quantity & Revenue are strongly correlated (0.76); Price & Revenue are weakly correlated (0.06).

- **Box Plots / Outlier Summary**
  - Files: `Week_01/figures/box_quantity.png`, `Week_01/figures/box_price.png`
  - Why: Visual outlier detection to understand returns, cancellations and large wholesale orders.
  - Insight: Quantity outliers ≈ 10.91%, Price outliers ≈ 6.38% — many reflect wholesale or cancelled orders rather than data entry errors.

## How to export plots from the notebook
In the notebook, after creating each Matplotlib/Seaborn plot, add a `savefig` call. Example (run inside the cell that draws the plot):

```python
# after your plotting code
plt.tight_layout()
plt.savefig('Week_01/figures/top10_qty.png', bbox_inches='tight', dpi=150)
plt.show()
```

Repeat for each plot using the file names listed above. Make sure the folder `Week_01/figures/` exists (create it if needed).

## Reproduce locally
1. Clone or download the repo.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install pandas numpy matplotlib seaborn openpyxl
```

3. Open `Week_01/Online_Retail_II_EDA.ipynb` in Jupyter or VS Code, run the cells and export the figures with the `savefig` snippet above.

## Notes & next steps
- Investigate and reduce the ~22.77% missing `CustomerID` rows (guest checkout, data issues) before building personalized recommenders.
- Handle outliers carefully: many are legitimate business transactions (B2B wholesale, cancellations).
- Optionally add `requirements.txt` for environment reproducibility and commit exported images under `Week_01/figures/`.

