import pandas as pd
# Loading the data
customer_activity_df = pd.read_csv('customer_activity_data.csv')
print(f"The shape of the DataFrame is: {customer_activity_df.shape}")
customer_activity_df.head()

customer_activity_df.info()

