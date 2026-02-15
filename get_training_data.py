import json
import csv
import random
from tqdm import tqdm
from pathlib import Path

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}")
    return data

def write_tsv(path, data):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(data)

def create_qampari_training_files():
    # Set random seed for reproducibility
    random.seed(42)
    
    train_data = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    print(f'Loaded {len(train_data)} training examples')
    
    # Create mapping from original index to new index (filtering empty queries)
    valid_indices = {}
    new_qid = 0
    
    # Create queries.tsv - only include queries with non-empty text
    queries_tsv = []
    empty_queries = []
    
    for i, item in enumerate(tqdm(train_data, desc="Creating queries")):
        query_text = item.get('question_text', '').strip()
        
        # Skip empty queries
        if not query_text:
            empty_queries.append(i)
            print(f'Skipping empty query at original index {i}')
            continue
        
        # Remove tabs, newlines from query text
        query_text = query_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        
        # Map old index to new index
        valid_indices[i] = new_qid
        queries_tsv.append([str(new_qid), query_text])
        new_qid += 1
    
    output_dir = Path('/scratch/dq2024/ColBERT/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_tsv(output_dir / 'qampari_train_queries.tsv', queries_tsv)
    print(f'\nCreated queries.tsv with {len(queries_tsv)} queries')
    print(f'Skipped {len(empty_queries)} queries with empty text: {empty_queries}')
    
    # Create triples.jsonl - only for valid queries
    triples = []
    queries_without_data = 0
    
    for i, item in enumerate(tqdm(train_data, desc="Creating triples")):
        # Skip if query was filtered out
        if i not in valid_indices:
            continue
        
        qid = valid_indices[i]  # Use new QID
        
        # Get positive passage IDs
        positive_pids = []
        for pos_ctx in item.get('positive_ctxs', []):
            pid = pos_ctx.get('id', '').strip()
            if pid and pid not in positive_pids:
                positive_pids.append(pid)
        
        # Get negative passage IDs
        negative_pids = []
        for neg_ctx in item.get('negative_ctxs', []):
            pid = neg_ctx.get('id', '').strip()
            if pid and pid not in negative_pids:
                negative_pids.append(pid)
        
        # Create triples
        if positive_pids and negative_pids:
            for pos_pid in positive_pids:
                neg_pid = random.choice(negative_pids)
                triples.append([qid, pos_pid, neg_pid])
        else:
            queries_without_data += 1
            if not positive_pids:
                print(f'Warning: Query {qid} (original {i}) has no positive passages')
            if not negative_pids:
                print(f'Warning: Query {qid} (original {i}) has no negative passages')
    
    # Write triples to JSONL
    with open(output_dir / 'qampari_train_triples.jsonl', 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    
    print(f'\nCreated triples.jsonl with {len(triples)} triples')
    print(f'Queries without sufficient data: {queries_without_data}')
    print(f'Average triples per valid query: {len(triples) / len(queries_tsv):.2f}')
    print(f'\nFiles created in: {output_dir}')
    print(f'  - qampari_train_queries.tsv ({len(queries_tsv)} queries)')
    print(f'  - qampari_train_triples.jsonl ({len(triples)} triples)')
    print(f'\nNote: Each positive passage is paired with a randomly sampled negative.')
    print(f'      In-batch negatives will provide 63 additional negatives during training.')

if __name__ == '__main__':
    create_qampari_training_files()