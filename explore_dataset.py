import pickle
import numpy as np

# Load the dataset
with open('dataset/naval_dataset.pkl', 'rb') as f:
    train_data, train_label, val_data, val_label = pickle.load(f)

# Print dataset information
print("\nDataset Information:")
print("-" * 50)
print(f"Training Data Shape: {train_data.shape}")
print(f"Training Labels Shape: {train_label.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Validation Labels Shape: {val_label.shape}")

# Print sample data
print("\nSample Training Data (first 2 samples):")
print("-" * 50)
print(train_data[:2])

print("\nSample Training Labels (first 2 samples):")
print("-" * 50)
print(train_label[:2])

# Convert to numpy arrays for easier viewing
train_data_np = train_data
train_label_np = train_label

print("\nTraining Data Statistics:")
print("-" * 50)
print(f"Min value: {np.min(train_data_np):.4f}")
print(f"Max value: {np.max(train_data_np):.4f}")
print(f"Mean value: {np.mean(train_data_np):.4f}")
print(f"Standard deviation: {np.std(train_data_np):.4f}")

# Check label distribution
unique_labels, label_counts = np.unique(train_label_np, return_counts=True)
print("\nLabel Distribution:")
print("-" * 50)
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} samples ({count/len(train_label_np)*100:.2f}%)")
