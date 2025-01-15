import pandas as pd

class DataFrameInfo:

    def __init__(self, df):
        self.df = df

    def describe_columns(self):
        """
        Describes all columns in the DataFrame, including data types.
        """
        print("Column Descriptions:")
        print(self.df.dtypes)

    def get_statistical_values(self, column=None):
        """
        Extracts statistical values (median, standard deviation, mean) for 
        a specified column or the entire DataFrame.

        Args:
            column (str, optional): Name of the column to extract statistics for. 
                                   If None, extracts for all numerical columns.
        """
        if column:
            if self.df[column].dtype in ['int64', 'float64']:
                print(f"Statistics for column '{column}':")
                print(f"Median: {self.df[column].median()}")
                print(f"Standard Deviation: {self.df[column].std()}")
                print(f"Mean: {self.df[column].mean()}")
            else:
                print(f"Column '{column}' is not numeric.")
        else:
            print("Statistics for all numeric columns:")
            for col in self.df.select_dtypes(include=['int64', 'float64']):
                print(f"Column '{col}':")
                print(f"Median: {self.df[col].median()}")
                print(f"Standard Deviation: {self.df[col].std()}")
                print(f"Mean: {self.df[col].mean()}")
                print("-" * 20)

    def count_distinct_values(self):
        """
        Counts distinct values in categorical columns.
        """
        print("Distinct Value Counts for Categorical Columns:")
        for col in self.df.select_dtypes(include=['object', 'category']):
            print(f"Column '{col}': {len(self.df[col].unique())} distinct values")

    def get_shape(self):
        """
        Prints the shape of the DataFrame (rows, columns).
        """
        print(f"DataFrame Shape: {self.df.shape}")

    def get_null_value_counts(self):
        """
        Generates count/percentage of NULL values in each column.
        """
        print("NULL Value Counts:")
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            null_percentage = (null_count / len(self.df)) * 100
            print(f"Column '{col}': {null_count} ({null_percentage:.2f}%) NULL values")

    def get_column_value_counts(self, column):
        """
        Prints the value counts for a given column.
        """
        print(f"Value Counts for Column '{column}':")
        print(self.df[column].value_counts())

# Example usage
# Assuming you have a DataFrame 'df'
df = pd.read_csv('customer_activity_data.csv') 

df_info = DataFrameInfo(df)
df_info.describe_columns()
df_info.get_statistical_values()
df_info.count_distinct_values()
df_info.get_shape()
df_info.get_null_value_counts()
df_info.get_column_value_counts('month')