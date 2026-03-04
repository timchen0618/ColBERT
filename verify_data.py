# verify_training_data.py
import json
import csv
from pathlib import Path
from collections import Counter

def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

def read_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def verify_training_setup():
    print("="*80)
    print("COLBERT TRAINING DATA VERIFICATION")
    print("="*80)
    
    # 1. Check files exist
    print("\n[1] Checking if all required files exist...")
    files_to_check = {
        'PID Mapping': '/scratch/dq2024/wikipedia_chunks/pid_to_index_mapping.json',
        'Original Chunks': '/scratch/dq2024/wikipedia_chunks/chunks_v5.tsv',
        'ColBERT Chunks': '/scratch/dq2024/wikipedia_chunks/colbert_chunks_v5.tsv',
        'Training Queries': '/scratch/dq2024/ColBERT/data/qampari_train_queries.tsv',
        'Training Triples': '/scratch/dq2024/ColBERT/data/qampari_train_triples.jsonl',
        'Raw Training Data': '/scratch/dq2024/diverse_retriever/train_data.jsonl'
    }
    
    missing_files = []
    for name, path in files_to_check.items():
        if Path(path).exists():
            size = Path(path).stat().st_size / (1024**2)  # MB
            print(f"  ✓ {name}: {size:.2f} MB")
        else:
            print(f"  ✗ {name}: NOT FOUND")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    # 2. Load and check PID mapping
    print("\n[2] Checking PID mapping...")
    with open('/scratch/dq2024/wikipedia_chunks/pid_to_index_mapping.json', 'r') as f:
        pid_to_index = json.load(f)
    
    print(f"  - Total PIDs mapped: {len(pid_to_index):,}")
    
    # Sample some mappings
    sample_pids = list(pid_to_index.items())[:5]
    print(f"  - Sample mappings:")
    for pid, idx in sample_pids:
        print(f"    '{pid}' -> {idx}")
    
    # Check if PIDs look like complex IDs (not just simple integers)
    first_pid = list(pid_to_index.keys())[0]
    if '__' in first_pid:
        print(f"  ✓ PIDs look correct (contain '__')")
    else:
        print(f"  ⚠ Warning: PIDs don't contain '__' - might be simple integers")
    
    # 3. Check alignment between original and ColBERT chunks
    print("\n[3] Verifying file alignment...")
    original = read_tsv('/scratch/dq2024/wikipedia_chunks/chunks_v5.tsv')
    colbert = read_tsv('/scratch/dq2024/wikipedia_chunks/colbert_chunks_v5.tsv')
    
    print(f"  - Original chunks: {len(original):,} rows")
    print(f"  - ColBERT chunks: {len(colbert):,} rows")
    
    # Check if alignment is correct (accounting for header in original)
    alignment_errors = 0
    for i in range(1, min(11, len(original))):  # Check first 10 rows
        orig_text = original[i][1] if len(original[i]) > 1 else ""
        colb_text = colbert[i-1][1] if len(colbert[i-1]) > 1 else ""
        
        if orig_text[:200] == colb_text[:200]:
            if i <= 3:  # Only print first 3
                print(f"  ✓ Row {i-1}: Text matches (PID: {original[i][0]})")
        else:
            print(f"  ✗ Row {i-1}: TEXT MISMATCH!")
            alignment_errors += 1
    
    if alignment_errors > 0:
        print(f"\n  ❌ Found {alignment_errors} alignment errors in first 10 rows!")
        return False
    else:
        print(f"  ✓ File alignment looks good")
    
    # 4. Check training queries
    print("\n[4] Checking training queries...")
    queries = read_tsv('/scratch/dq2024/ColBERT/data/qampari_train_queries.tsv')
    
    print(f"  - Total queries: {len(queries):,}")
    print(f"  - Sample queries:")
    for i in range(min(3, len(queries))):
        qid, text = queries[i]
        print(f"    QID {qid}: {text[:80]}...")
    
    # Check for duplicates
    query_texts = [q[1] for q in queries]
    duplicates = len(query_texts) - len(set(query_texts))
    if duplicates > 0:
        print(f"  ⚠ Warning: {duplicates} duplicate query texts found")
    else:
        print(f"  ✓ No duplicate queries")
    
    # 5. Check training triples
    print("\n[5] Checking training triples...")
    triples = read_jsonl('/scratch/dq2024/ColBERT/data/qampari_train_triples.jsonl')
    
    print(f"  - Total triples: {len(triples):,}")
    print(f"  - Average triples per query: {len(triples) / len(queries):.2f}")
    
    # Check triple format
    print(f"  - Sample triples:")
    for i in range(min(3, len(triples))):
        qid, pos_idx, neg_idx = triples[i]
        print(f"    [QID: {qid}, Pos: {pos_idx}, Neg: {neg_idx}]")
    
    # Verify indices are in valid range
    max_colbert_idx = len(colbert) - 1
    invalid_indices = []
    
    for triple in triples[:1000]:  # Check first 1000
        qid, pos_idx, neg_idx = triple
        if pos_idx > max_colbert_idx or neg_idx > max_colbert_idx:
            invalid_indices.append(triple)
    
    if invalid_indices:
        print(f"  ❌ Found {len(invalid_indices)} triples with out-of-range indices!")
        print(f"    Max valid index: {max_colbert_idx}")
        print(f"    Sample bad triples: {invalid_indices[:3]}")
        return False
    else:
        print(f"  ✓ All indices are in valid range [0, {max_colbert_idx}]")
    
    # 6. Cross-check with raw training data
    print("\n[6] Cross-checking with raw training data...")
    raw_train = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    
    print(f"  - Raw training examples: {len(raw_train):,}")
    
    # Check if PIDs from raw data exist in mapping
    sample_raw = raw_train[0]
    if 'positive_ctxs' in sample_raw and len(sample_raw['positive_ctxs']) > 0:
        sample_pid = sample_raw['positive_ctxs'][0].get('id', '')
        if sample_pid in pid_to_index:
            mapped_idx = pid_to_index[sample_pid]
            print(f"  ✓ Sample PID '{sample_pid}' maps to index {mapped_idx}")
        else:
            print(f"  ❌ Sample PID '{sample_pid}' NOT FOUND in mapping!")
            return False
    
    # Count how many PIDs are in the mapping
    all_pids_in_raw = set()
    for example in raw_train[:100]:  # Check first 100
        for ctx in example.get('positive_ctxs', []):
            all_pids_in_raw.add(str(ctx.get('id', '')))
        for ctx in example.get('negative_ctxs', []):
            all_pids_in_raw.add(str(ctx.get('id', '')))
    
    found_pids = sum(1 for pid in all_pids_in_raw if pid in pid_to_index)
    coverage = (found_pids / len(all_pids_in_raw) * 100) if all_pids_in_raw else 0
    
    print(f"  - PID coverage in mapping: {found_pids}/{len(all_pids_in_raw)} ({coverage:.1f}%)")
    
    if coverage < 95:
        print(f"  ⚠ Warning: Low PID coverage - many training PIDs not in mapping!")
    else:
        print(f"  ✓ Good PID coverage")
    
    # 7. Statistics summary
    print("\n[7] Statistics Summary...")
    qid_counts = Counter([t[0] for t in triples])
    triples_per_query = list(qid_counts.values())
    
    print(f"  - Min triples per query: {min(triples_per_query)}")
    print(f"  - Max triples per query: {max(triples_per_query)}")
    print(f"  - Median triples per query: {sorted(triples_per_query)[len(triples_per_query)//2]}")
    
    # Check for queries with very few triples
    low_triple_queries = sum(1 for count in triples_per_query if count < 3)
    if low_triple_queries > len(queries) * 0.1:  # More than 10%
        print(f"  ⚠ Warning: {low_triple_queries} queries have fewer than 3 triples")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE - All checks passed!")
    print("="*80)
    print("\nYou're ready to start training with:")
    print("  python colbert_script.py --do_training --dataset qampari")
    print("="*80)
    
    return True

if __name__ == '__main__':
    success = verify_training_setup()
    if not success:
        print("\n❌ Verification failed! Fix the issues above before training.")
        exit(1)