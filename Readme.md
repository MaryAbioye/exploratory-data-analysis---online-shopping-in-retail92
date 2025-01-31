# Project Title: AWS RDS Data Handling with Python

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage Instructions](#usage-instructions)
4. [File Structure](#file-structure)
5. [License](#license)

## Project Description
This project demonstrates how to connect to an AWS RDS database, extract data using Python, and perform essential data manipulation and storage operations. The project aims to enhance my skills in database handling, data engineering, and Python programming.

### Key Features:
- Establishing a connection to AWS RDS using SQLAlchemy.
- Extracting data from the `customer_activity` table into Pandas DataFrames.
- Storing data locally in `.csv` format for efficient analysis.
- Loading data into a DataFrame for further exploration and visualization.

### What I Learned:
- Handling database credentials securely with YAML files.
- Using SQLAlchemy to manage database connections.
- Saving and loading data efficiently with Pandas.
- Analysing and transforming data

###Usage
Python files:
db_utils.py:
Uses Sqlalchemy to connect to remote database and creates .csv file of downloaded data.

DataFrameinfo.py:
Class used to generate basic info about a dataframe, including data types, descriptive statistics, df shape and null values.

plotter.py:
Contains two classes, one to perform transformations on the data and the second to perform transformations on the dataframe.
Data transformations include changing the data type.
Dataframe transformations include removal of null-values (drop columns/rows), imputing data with mean/median/mode, and performing transformations to correct skewed data.

plotter.py:
Contains a class used to generate plots to visualize a dataset for statistical analysis.
Visualisations include, visualisation of null-values, bar chart, histogram, heatmaps, boxplots, etc.
Also contains statistical tests including the chi squared test, and k squared test.
Additionally, there are methods to generate visuals to assess normalisation of data to correct for skew.


Jupyter Notebooks:
EDA_Online_Shopping_Retail.ipynb:
Contains the workflow of the initial EDA process including data extraction, loading and cleaning/transformations.
Key steps included:
Correcting data types.
Handling missing values.
Obtaining basic info on df.
Checking and correcting distribution.
Handling outliers.
Checking for overly correlated data.

Analysis_Answering Business Questions.ipynb:
Contains the workflow for a more detailed analysis and visualisation of the data.
Key questions addressed:
Where are our customers spending their time?
What software are our customers using?
What factors are influencing revenue?


###File structure
AiCoreEDA_Project
├── db_utils.py
├── DataFrameinfo.py
├── DataFrameTransform.py
├── plotter.py
├── EDA_Online_Shopping_Retail.ipynb
├── Analysis_Answering Business Questions.ipynb
└── README.md


## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/MaryAbioye/exploratory-data-analysis---online-shopping-in-retail92.git
