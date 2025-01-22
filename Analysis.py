import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotter import Plotter
from dataframe_transform import DataFrameTransform

# Load the dataset
data_file = "customer_activity_data.csv"
df = pd.read_csv(data_file)

# Initialize transformation and plotting objects
df_transform = DataFrameTransform(df)
plotter = Plotter(df)

# Handle NULLs and save the cleaned DataFrame
print("\n=== Checking and Handling NULLs ===")
null_summary = df_transform.check_nulls()
df_transform.impute_nulls(strategy="mean")
df_cleaned = df.copy()
df_cleaned.to_csv("cleaned_website_data.csv", index=False)

# Question 1: Are sales proportionally happening more on weekends?
df_cleaned['DayOfWeek'] = pd.to_datetime(df_cleaned['Date']).dt.day_name()
day_sales = df_cleaned.groupby('DayOfWeek')['Sales'].sum().sort_index(key=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(x))
print("\n=== Sales Distribution by Day of Week ===")
print(day_sales)
plotter.plot_box('Sales')

# Question 2: Which regions are generating the most revenue currently?
region_revenue = df_cleaned.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print("\n=== Revenue by Region ===")
print(region_revenue)

# Question 3: Is there any particular website traffic that stands out when generating sales?
traffic_sales = df_cleaned.groupby('TrafficSource')['Sales'].sum().sort_values(ascending=False)
print("\n=== Sales by Traffic Source ===")
print(traffic_sales)

# Question 4: What percentage of time is spent on the website performing administrative/product or informational related tasks?
time_columns = ['AdminTaskTime', 'ProductTaskTime', 'InfoTaskTime']
total_time = df_cleaned[time_columns].sum().sum()
time_percentages = (df_cleaned[time_columns].sum() / total_time) * 100
print("\n=== Percentage of Time Spent on Tasks ===")
print(time_percentages)

# Question 5: Are there any informational/administrative tasks which users spend time doing most?
admin_task_time = df_cleaned['AdminTaskTime'].sum()
info_task_time = df_cleaned['InfoTaskTime'].sum()
print("\n=== Total Time for Tasks ===")
print(f"Administrative Task Time: {admin_task_time}\nInformational Task Time: {info_task_time}")

# Question 6: What is the breakdown of months making the most sales?
df_cleaned['Month'] = pd.to_datetime(df_cleaned['Date']).dt.month_name()
month_sales = df_cleaned.groupby('Month')['Sales'].sum().sort_values(ascending=False)
print("\n=== Sales Breakdown by Month ===")
print(month_sales)

# Save final outputs to CSV
analysis_output = {
    'DaySales': day_sales,
    'RegionRevenue': region_revenue,
    'TrafficSales': traffic_sales,
    'TimePercentages': time_percentages,
    'MonthSales': month_sales
}
for key, data in analysis_output.items():
    data.to_csv(f"{key.lower()}_analysis.csv", index=True)
