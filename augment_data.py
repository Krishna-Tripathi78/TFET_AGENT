import pandas as pd
import numpy as np

# Read original data
df = pd.read_csv('c:/Users/krish/TFET _ AGENT/aluminum_tfet_data.csv')

print(f"Original samples: {len(df)}")

# Generate 100 samples using statistical augmentation
augmented_data = []

for _ in range(100):
    # Randomly select a base sample
    base_idx = np.random.randint(0, len(df))
    base_row = df.iloc[base_idx]
    
    # Add small random variations (±5%)
    new_row = {
        'gate_voltage': base_row['gate_voltage'] * (1 + np.random.uniform(-0.05, 0.05)),
        'drain_voltage': base_row['drain_voltage'] * (1 + np.random.uniform(-0.05, 0.05)),
        'channel_length': base_row['channel_length'] * (1 + np.random.uniform(-0.05, 0.05)),
        'oxide_thickness': base_row['oxide_thickness'] * (1 + np.random.uniform(-0.05, 0.05)),
        'work_function': base_row['work_function'],
        'contact_resistance': base_row['contact_resistance'] * (1 + np.random.uniform(-0.05, 0.05))
    }
    
    augmented_data.append(new_row)

# Create augmented dataframe
augmented_df = pd.DataFrame(augmented_data)

# Save
output_path = 'c:/Users/krish/TFET _ AGENT/aluminum_tfet_data_augmented.csv'
augmented_df.to_csv(output_path, index=False)

print(f"Augmented samples: {len(augmented_df)}")
print(f"Saved to: {output_path}")
print("\nSample statistics:")
print(augmented_df.describe())
