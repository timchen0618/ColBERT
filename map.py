# create_pid_mapping.py
import csv
import json
from tqdm import tqdm

def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

# Load original chunks file
print('Loading original chunks file...')
corpus = read_tsv('/scratch/dq2024/wikipedia_chunks/chunks_v5.tsv')
print(f'Loaded {len(corpus)} passages')

# Create mapping from complex PID to integer index
pid_to_index = {}

for i, row in enumerate(tqdm(corpus, desc="Building PID mapping")):
    if i == 0:  # Skip header row
        continue
    
    if len(row) >= 2:
        complex_pid = row[0]  # e.g., "32374689__0"
        index = i - 1  # Row 1 -> index 0, Row 2 -> index 1, etc.
        pid_to_index[complex_pid] = index

print(f'Created mapping for {len(pid_to_index)} passages')

# Save the mapping
output_path = '/scratch/dq2024/wikipedia_chunks/pid_to_index_mapping.json'
with open(output_path, 'w') as f:
    json.dump(pid_to_index, f)

print(f'Saved PID mapping to {output_path}')

# Verify a few entries
print('\nSample mappings:')
for i, (pid, idx) in enumerate(list(pid_to_index.items())[:5]):
    print(f'  {pid} -> {idx}')