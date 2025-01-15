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

    def summary(self) -> None:
        """
        Prints a summary of the DataFrame including data types and missing values.
        """
        print("DataFrame Summary:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())


# Example usage (for understanding only):
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "date_column": ["2023-01-01", "2023-02-01", "not_a_date"],
        "numeric_column": ["100", "200.5", "invalid"],
        "text_column": ["$100", "$200", "$300"],
        "category_column": ["A", "B", "A"],
    }
    df = pd.DataFrame(data)

    transformer = DataTransform(df)

    transformer.convert_to_datetime("date_column")
    transformer.convert_to_numeric("numeric_column")
    transformer.strip_symbols("text_column", symbols=["$"])
    transformer.convert_to_categorical("category_column")

    transformer.summary()