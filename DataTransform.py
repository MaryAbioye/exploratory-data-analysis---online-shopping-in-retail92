# Milestone 3 Task 1
import pandas as pd

class DataTransform:
    """
    A class to handle data transformations for a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataTransform class with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be transformed.
        """
        self.df = df

    def convert_to_datetime(self, column: str, date_format: str = None) -> None:
        """
        Converts a column to datetime format.

        Args:
            column (str): The name of the column to convert.
            date_format (str, optional): The format of the date in the column. Defaults to None.
        """
        try:
            self.df[column] = pd.to_datetime(self.df[column], format=date_format, errors="coerce")
            print(f"Column '{column}' successfully converted to datetime.")
        except Exception as e:
            print(f"Error converting column '{column}' to datetime: {e}")

    def convert_to_numeric(self, column: str) -> None:
        """
        Converts a column to numeric format.

        Args:
            column (str): The name of the column to convert.
        """
        try:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
            print(f"Column '{column}' successfully converted to numeric.")
        except Exception as e:
            print(f"Error converting column '{column}' to numeric: {e}")

    def strip_symbols(self, column: str, symbols: list) -> None:
        """
        Removes specified symbols from a column.

        Args:
            column (str): The name of the column to clean.
            symbols (list): A list of symbols to remove.
        """
        try:
            for symbol in symbols:
                self.df[column] = self.df[column].str.replace(symbol, "", regex=True)
            print(f"Symbols {symbols} removed from column '{column}'.")
        except Exception as e:
            print(f"Error removing symbols from column '{column}': {e}")


    def to_category(self, *columns):
        """
        Converts specified columns to categorical data type.

        Args:
            *columns: Column names to be converted to categorical.
        """
        for column in columns:
            try:
                self.df[column] = self.df[column].astype("category")
                print(f"Column '{column}' successfully converted to categorical.")
            except Exception as e:
                print(f"Error converting column '{column}' to categorical: {e}")

    def to_Int(self, *columns):
        """
        Converts specified columns to integer format, coercing errors to NaN.

        Args:
            *columns: Column names to be converted to integer.
        """
        for column in columns:
            try:
                self.df[column] = pd.to_numeric(self.df[column], errors="coerce").astype("Int64")
                print(f"Column '{column}' successfully converted to integer.")
            except Exception as e:
                print(f"Error converting column '{column}' to integer: {e}")

    def summary(self) -> None:
        """
        Prints a summary of the DataFrame including data types and missing values.
        """
        print("DataFrame Summary:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
