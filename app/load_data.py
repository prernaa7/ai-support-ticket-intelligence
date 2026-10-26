
# Import Pandas to work with tabular data
import pandas as pd

# Define the location of the dataset
DATA_PATH = "data/tickets.csv"

# Load the CSV file into a DataFrame
tickets = pd.read_csv(DATA_PATH)

# Display the first five support tickets
print("\nFirst five tickets:")
print(tickets.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(tickets.shape)

# Display column names and data types
print("\nDataset information:")
tickets.info()

# Count tickets in each category
print("\nCategory distribution:")
print(tickets["category"].value_counts())

# Count tickets by priority
print("\nPriority distribution:")
print(tickets["priority"].value_counts())

# Check for missing values
print("\nMissing values:")
print(tickets.isnull().sum())
