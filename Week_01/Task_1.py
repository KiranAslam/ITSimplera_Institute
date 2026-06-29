import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style for clean visuals
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. LOAD THE DATASET & DISPLAY BASIC INFO

file_path = "Data/online_retail_II.xlsx"
print("Loading data...")

df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

print("\n--- Basic Information ---")
print(f"Dataset Shape: {df.shape}")
print("\nData Types and Missing Values:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Add a Revenue column for subsequent analysis

df['Revenue'] = df['Quantity'] * df['Price']

# 2. IDENTIFY MISSING VALUES AND DUPLICATE ROWS

print("\n--- Data Quality Observations ---")
missing_vals = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_table = pd.DataFrame({'Missing Values': missing_vals, 'Percentage (%)': missing_percent})
print("Missing Values Per Column:")
print(missing_table)

duplicate_count = df.duplicated().sum()
print(f"\nTotal Duplicate Rows: {duplicate_count} ({duplicate_count / len(df) * 100:.2f}%)")

# 3. TOP 10 BEST-SELLING PRODUCTS (QUANTITY & REVENUE)
# Group by Description (ignoring missing descriptions for top selling calculations)

top_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_rev = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

# Plotting Top 10 by Quantity
plt.figure()
sns.barplot(x=top_qty.values, y=top_qty.index, palette="Blues_r")
plt.title("Top 10 Best-Selling Products by Quantity (2009-2011)", fontsize=14, pad=15)
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Description")
plt.tight_layout()
plt.show()

# Plotting Top 10 by Revenue
plt.figure()
sns.barplot(x=top_rev.values, y=top_rev.index, palette="Greens_r")
plt.title("Top 10 Products by Total Revenue (2009-2011)", fontsize=14, pad=15)
plt.xlabel("Total Revenue (£)")
plt.ylabel("Product Description")
plt.tight_layout()
plt.show()


# 4. SALES PERFORMANCE BY COUNTRY
country_performance = df.groupby('Country').agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Revenue=('Revenue', 'sum')
).sort_values(by='Total_Revenue', ascending=False)

print("\n--- Top 5 Countries by Revenue ---")
print(country_performance.head(5))

# Plot Top Countries 
plt.figure()
sns.barplot(x=country_performance.head(10)['Total_Revenue'], y=country_performance.head(10).index, palette="Purples_r")
plt.title("Top 10 Countries by Revenue (2009-2011)", fontsize=14, pad=15)
plt.xlabel("Total Revenue (£)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# 5. REVENUE OVER TIME (MONTHLY TREND)
# Extract Year-Month for grouping
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().to_timestamp()

plt.figure()
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', color='crimson', linewidth=2)
plt.title("Monthly Revenue Trend (2009-2011)", fontsize=14, pad=15)
plt.xlabel("Timeline")
plt.ylabel("Revenue (£)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# 6. CORRELATION HEATMAP
plt.figure()
# Filtering numerical columns
numerical_cols = df[['Quantity', 'Price', 'Customer ID', 'Revenue']]
sns.heatmap(numerical_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Variables", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# 7. OUTLIER DETECTION VIA BOX PLOTS
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, y='Quantity', ax=axes[0], color='skyblue')
axes[0].set_title("Box Plot of Quantity (Outlier Detection)")
axes[0].set_ylabel("Quantity")

sns.boxplot(data=df, y='Price', ax=axes[1], color='salmon')
axes[1].set_title("Box Plot of Price (Outlier Detection)")
axes[1].set_ylabel("Unit Price (£)")

plt.suptitle("Outlier Analysis (Untreated Data)", fontsize=16)
plt.tight_layout()
plt.show()