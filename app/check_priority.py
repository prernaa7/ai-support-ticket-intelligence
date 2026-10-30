
import pandas as pd

# Load dataset
df = pd.read_csv("data/tickets.csv")

print("Dataset shape:", df.shape)

# Display priority distribution
print("\nPriority Distribution:")
print(df["priority"].value_counts())

# Check missing priorities
print("\nMissing priorities:")
print(df["priority"].isnull().sum())

# Display ticket descriptions and priorities
print("\nTicket Priority Details:")
print(df[["description", "priority"]].to_string(index=False))
