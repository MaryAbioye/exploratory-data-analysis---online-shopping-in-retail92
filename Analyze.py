import pandas as pd
import numpy as np
from Plotter import Plotter, DataFrameTransform

# Load the data
data_path = 'customer_activity_data.csv'  # Update with the path to your dataset
df = pd.read_csv(data_path)

# Initialize classes
transformer = DataFrameTransform(df)
plotter = Plotter(df)

# Handle null values and save the cleaned DataFrame
null_summary = transformer.check_nulls()
transformer.impute_nulls(strategy="mean")  # Impute nulls

cleaned_data_path = 'cleaned_dataset.csv'
df.to_csv(cleaned_data_path, index=False)  # Save cleaned data for further use

# Start answering questions
# Question 1: Are sales proportionally happening more on weekends?
print(f"3-TESTING")
df['Day_of_Week'] = pd.to_datetime(df['sale_date']).dt.day_name()
print(f"1 TESTING")
weekend_sales = df[df['Day_of_Week'].isin(['Saturday', 'Sunday'])]['sales'].sum()
print(f"2 TESTING")
total_sales = df['sales'].sum()
print(f"3 TESTING")
weekend_sales_percentage = (weekend_sales / total_sales) * 100
print(f"Weekend sales contribute {weekend_sales_percentage:.2f}% of total sales.")

# Question 2: Which regions are generating the most revenue currently?
region_revenue = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
print("Revenue by region:")
print(region_revenue)

# Question 3: Is there any particular website traffic that stands out when generating sales?
traffic_sales = df.groupby('traffic_source')['sales'].sum().sort_values(ascending=False)
print("Sales by traffic source:")
print(traffic_sales)

# Question 4: What percentage of time is spent on the website performing administrative/product or informational related tasks?
admin_time = df['administrative_duration'].sum()
product_time = df['product_duration'].sum()
info_time = df['informational_duration'].sum()
total_time = admin_time + product_time + info_time

admin_percentage = (admin_time / total_time) * 100
product_percentage = (product_time / total_time) * 100
info_percentage = (info_time / total_time) * 100

print("Time spent breakdown:")
print(f"Administrative: {admin_percentage:.2f}%")
print(f"Product: {product_percentage:.2f}%")
print(f"Informational: {info_percentage:.2f}%")

# Question 5: Are there any informational/administrative tasks which users spend time doing most?
admin_tasks = df.groupby('administrative_task')['administrative_duration'].sum().sort_values(ascending=False)
info_tasks = df.groupby('informational_task')['informational_duration'].sum().sort_values(ascending=False)

print("Time spent on administrative tasks:")
print(admin_tasks)
print("Time spent on informational tasks:")
print(info_tasks)

# Question 6: What is the breakdown of months making the most sales?
df['Month'] = pd.to_datetime(df['sale_date']).dt.month_name()
monthly_sales = df.groupby('Month')['sales'].sum().sort_values(ascending=False)
print("Sales by month:")
print(monthly_sales)
