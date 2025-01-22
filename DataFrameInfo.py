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


# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "date_column": ["2023-01-01", "2023-02-01", "not_a_date"],
        "numeric_column": ["100", "200.5", "invalid"],
        "text_column": ["$100", "$200", "$300"],
        "category_column": ["A", "B", "A"],
    }
    df = pd.DataFrame(data)

    print("Initial DataFrame:")
    print(df)

    # Data Transformation
    transformer = DataTransform(df)
    transformer.convert_to_datetime("date_column")
    transformer.convert_to_numeric("numeric_column")
    transformer.strip_symbols("text_column", symbols=["$"])
    transformer.to_category('traffic_type', 'operating_systems', 'browser', 'region', 'visitor_type')
    transformer.to_Int('administrative','product_related')
    

    print("\nTransformed DataFrame:")
    print(df)

    # Data Analysis
    info = DataFrameInfo(df)
    info.describe_columns()
    info.get_statistics()
    info.count_distinct()
    info.count_nulls()
    info.display_shape()
df.info()