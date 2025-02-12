import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "../Res/BGL_sequence_mapped.csv"
data = pd.read_csv(file_path)

# Split data: 80% for training, 20% for testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Save the split files
train_data.to_csv('../Res/BGL_sequence_mapped_train.csv', index=False)
test_data.to_csv('../Res/BGL_sequence_mapped_test.csv', index=False)

print("Data has been split into train and test sets.")
