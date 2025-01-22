import pandas as pd


class DataTransformation:
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

    def convert_to_datetime(self, column: str, format: str = None) -> None:
        """
        Converts a column to datetime format.

        Args:
            column (str): The name of the column to convert.
            format (str, optional): The format of the date in the column. Defaults to None.
        """
        try:
            self.df[column] = pd.to_datetime(self.df[column], format=format, errors="coerce")
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

    def convert_to_categorical(self, column: str) -> None:
        """
        Converts a column to a categorical data type.

        Args:
            column (str): The name of the column to convert.
        """
        try:
            self.df[column] = self.df[column].astype("category")
            print(f"Column '{column}' successfully converted to categorical.")
        except Exception as e:
            print(f"Error converting column '{column}' to categorical: {e}")

    def convert_month_to_numeric(self, column: str) -> None:
        """
        Converts a month column (e.g., 'Jan', 'Feb') to numeric format.

        Args:
            column (str): The name of the column to convert.
        """
        try:
            months = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12,
            }
            self.df[column] = self.df[column].map(months)
            print(f"Column '{column}' successfully converted to numeric representation of months.")
        except Exception as e:
            print(f"Error converting column '{column}' to numeric: {e}")

    def summary(self) -> None:
        """
        Prints a summary of the DataFrame including data types and missing values.
        """
        print("DataFrame Summary:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
