# #milestone 3 task 2
import pandas as pd
from DataTransform import DataTransform



class DataFrameInfo:
    """
    A class to extract and summarize information about a Pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataFrameInfo class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df

    def describe_columns(self):
        """
        Describes the columns in the DataFrame, including their data types and non-null counts.
        """
        print("Column Description:")
        print(self.df.info())

    def get_statistics(self):
        """
        Extracts mean, median, and standard deviation for numerical columns.
        """
        stats = self.df.describe().T
        stats['median'] = self.df.median(numeric_only=True)
        print("Statistical Summary:")
        print(stats)

    def count_distinct(self):
        """
        Counts the distinct values in categorical columns.
        """
        print("Distinct Value Counts for Categorical Columns:")
        for column in self.df.select_dtypes(include=['category']).columns:
            print(f"{column}: {self.df[column].nunique()} unique values")

    def count_nulls(self):
        """
        Counts and calculates the percentage of NULL values in each column.
        """
        null_counts = self.df.isnull().sum()
        null_percentages = (self.df.isnull().mean()) * 100
        print("Null Value Counts and Percentages:")
        print(pd.DataFrame({"Count": null_counts, "Percentage": null_percentages}))

    def display_shape(self):
        """
        Prints the shape of the DataFrame.
        """
        print(f"DataFrame Shape: {self.df.shape}")


