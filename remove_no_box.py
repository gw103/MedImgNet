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

def label_list_ok(row):
    labels = [l.strip() for l in row.split('|')]
    return (labels == ['No Finding']) or all(l in allowed_labels for l in labels)

df_filtered = df[df['Finding Labels'].apply(label_list_ok)]

# Then sample 10% of No Finding rows
only_nf = df_filtered[df_filtered['Finding Labels'] == 'No Finding']
others = df_filtered[df_filtered['Finding Labels'] != 'No Finding']
df_final = pd.concat([
    others,
    only_nf.sample(frac=0.1, random_state=42)
])
