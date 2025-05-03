import pandas as pd

# Config
csv_path = "/home/gezhiwang103_gmail_com/datasets/nih-chest-xrays/Data_Entry_2017_cleaned.csv"
save_path = "/home/gezhiwang103_gmail_com/datasets/nih-chest-xrays/Data_Entry_filtered.csv"

# Only allow these labels
allowed_labels = {
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'No Finding'
}

# Load CSV
df = pd.read_csv(csv_path)

def is_valid_row(row):
    labels = [label.strip() for label in row['Finding Labels'].split('|')]
    if labels == ['No Finding']:
        return 'No Finding'
    return all(label in allowed_labels for label in labels)

# Apply filter
df['keep_flag'] = df.apply(is_valid_row, axis=1)

# Keep rows that are valid or 10% of 'No Finding'
valid_rows = df[df['keep_flag'] != 'No Finding']
no_finding_sample = df[df['keep_flag'] == 'No Finding'].sample(frac=0.1, random_state=42)

# Combine and save
filtered_df = pd.concat([valid_rows, no_finding_sample]).drop(columns=['keep_flag'])
filtered_df.to_csv(save_path, index=False)

print(f"✅ Saved filtered CSV to: {save_path}")
print(f"Remaining rows: {len(filtered_df)}")
