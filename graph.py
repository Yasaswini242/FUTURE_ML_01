import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/Sample - Superstore.csv", encoding="latin1")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Sort by date
df = df.sort_values("Order Date")

# Group sales by date
daily_sales = df.groupby("Order Date")["Sales"].sum()

# Plot graph
plt.figure(figsize=(12,6))
plt.plot(daily_sales)

plt.title("Daily Sales Trend")
plt.xlabel("Order Date")
plt.ylabel("Sales")

# Save graph
plt.savefig("sales_forecast.png")

# Show graph
plt.show()