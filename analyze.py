import json
from tqdm import tqdm

def read_jsonl(path):
    data = []
    malformed_count = 0
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_count += 1
                print(f'Warning: Skipping malformed JSON at line {line_num}')
    
    if malformed_count > 0:
        print(f'Total malformed lines skipped: {malformed_count}')
    
    return data

def analyze_qampari_data():
    train_data = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    print(f'Loaded {len(train_data)} training examples\n')
    
    total_positives = 0
    total_negatives = 0
    total_ground_truths = 0
    total_unique_positives = 0
    
    positives_per_query = []
    negatives_per_query = []
    
    queries_with_no_negatives = 0
    queries_with_no_positives = 0
    
    for item in tqdm(train_data, desc="Analyzing data"):
        # Count positive_ctxs
        positive_pids = []
        for pos_ctx in item.get('positive_ctxs', []):
            pid = pos_ctx['id']
            if pid not in positive_pids:
                positive_pids.append(pid)
        
        num_pos_raw = len(item.get('positive_ctxs', []))
        num_pos_unique = len(positive_pids)
        
        # Count negative_ctxs
        negative_pids = []
        for neg_ctx in item.get('negative_ctxs', []):
            pid = neg_ctx['id']
            if pid not in negative_pids:
                negative_pids.append(pid)
        
        num_neg = len(negative_pids)
        
        # Count ground_truths
        num_gt = len(item.get('ground_truths', []))
        
        total_positives += num_pos_raw
        total_unique_positives += num_pos_unique
        total_negatives += num_neg
        total_ground_truths += num_gt
        
        positives_per_query.append(num_pos_raw)
        negatives_per_query.append(num_neg)
        
        if num_pos_unique > 0 and num_neg == 0:
            queries_with_no_negatives += 1
        if num_pos_unique == 0:
            queries_with_no_positives += 1
    
    print(f'\n{"="*60}')
    print(f'QAMPARI Training Data Analysis')
    print(f'{"="*60}\n')
    
    print(f'Dataset Size:')
    print(f'  - Total queries: {len(train_data):,}')
    print(f'  - Total triples expected: {total_unique_positives:,}\n')
    
    print(f'Positive Passages (positive_ctxs):')
    print(f'  - Total positive_ctxs (raw): {total_positives:,}')
    print(f'  - Total unique positives: {total_unique_positives:,}')
    print(f'  - Average positive_ctxs per query: {total_positives / len(train_data):.2f}')
    print(f'  - Average unique positives per query: {total_unique_positives / len(train_data):.2f}')
    print(f'  - Min positives per query: {min(positives_per_query)}')
    print(f'  - Max positives per query: {max(positives_per_query)}\n')
    
    print(f'Negative Passages (negative_ctxs):')
    print(f'  - Total negatives: {total_negatives:,}')
    print(f'  - Average negatives per query: {total_negatives / len(train_data):.2f}')
    print(f'  - Min negatives per query: {min(negatives_per_query)}')
    print(f'  - Max negatives per query: {max(negatives_per_query)}\n')
    
    print(f'Ground Truths:')
    print(f'  - Total ground_truths: {total_ground_truths:,}')
    print(f'  - Average ground_truths per query: {total_ground_truths / len(train_data):.2f}\n')
    
    print(f'Data Quality:')
    print(f'  - Queries with positives but NO negatives: {queries_with_no_negatives}')
    print(f'  - Queries with NO positives: {queries_with_no_positives}')
    print(f'  - Duplicate positives filtered: {total_positives - total_unique_positives}\n')
    
    print(f'Expected vs Actual Triples:')
    print(f'  - Expected triples (unique positives): {total_unique_positives:,}')
    print(f'  - Actual triples created: 553,633')
    print(f'  - Difference: {total_unique_positives - 553633:,}')

if __name__ == '__main__':
    analyze_qampari_data()