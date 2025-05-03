import pandas as pd

from collections import Counter

df = pd.read_csv("../datasets/nih-chest-xrays/Data_Entry_filtered.csv")

# Split labels and flatten the list
all_labels = df['Finding Labels'].str.split('|').explode().str.strip()

# Count occurrences
label_counts = all_labels.value_counts()

# Show top 50 labels
print(label_counts.head(50))