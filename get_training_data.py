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
    random.seed(42)
    
    # Load PID to index mapping
    print('Loading PID to index mapping...')
    with open('/scratch/dq2024/wikipedia_chunks/pid_to_index_mapping.json', 'r') as f:
        pid_to_index = json.load(f)
    print(f'Loaded mapping for {len(pid_to_index)} passages')
    
    train_data = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    print(f'Loaded {len(train_data)} training examples')
    
    # Create queries
    valid_indices = {}
    new_qid = 0
    queries_tsv = []
    empty_queries = []
    
    for i, item in enumerate(tqdm(train_data, desc="Creating queries")):
        query_text = item.get('question_text', '').strip()
        
        if not query_text:
            empty_queries.append(i)
            continue
        
        query_text = query_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        valid_indices[i] = new_qid
        queries_tsv.append([str(new_qid), query_text])
        new_qid += 1
    
    output_dir = Path('/scratch/dq2024/ColBERT/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_tsv(output_dir / 'qampari_train_queries.tsv', queries_tsv)
    print(f'\nCreated queries.tsv with {len(queries_tsv)} queries')
    print(f'Skipped {len(empty_queries)} queries with empty text')
    
    # Create triples with INTEGER indices
    triples = []
    queries_without_data = 0
    missing_pids = set()
    
    for i, item in enumerate(tqdm(train_data, desc="Creating triples")):
        if i not in valid_indices:
            continue
        
        qid = valid_indices[i]
        
        # Convert string PIDs to integer indices
        positive_indices = []
        for pos_ctx in item.get('positive_ctxs', []):
            pid = str(pos_ctx.get('id', '')).strip()
            if pid in pid_to_index:
                idx = pid_to_index[pid]
                if idx not in positive_indices:
                    positive_indices.append(idx)
            elif pid:
                missing_pids.add(pid)
        
        negative_indices = []
        for neg_ctx in item.get('negative_ctxs', []):
            pid = str(neg_ctx.get('id', '')).strip()
            if pid in pid_to_index:
                idx = pid_to_index[pid]
                if idx not in negative_indices:
                    negative_indices.append(idx)
            elif pid:
                missing_pids.add(pid)
        
        # Create triples
        if positive_indices and negative_indices:
            for pos_idx in positive_indices:
                neg_idx = random.choice(negative_indices)
                triples.append([qid, pos_idx, neg_idx])
        else:
            queries_without_data += 1
    
    # Write triples
    with open(output_dir / 'qampari_train_triples.jsonl', 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    
    print(f'\nCreated triples.jsonl with {len(triples)} triples')
    print(f'Queries without sufficient data: {queries_without_data}')
    print(f'Average triples per valid query: {len(triples) / len(queries_tsv):.2f}')
    if missing_pids:
        print(f'\nWarning: {len(missing_pids)} PIDs not found in collection')
        print(f'First 10 missing PIDs: {list(missing_pids)[:10]}')
    print(f'\nFiles created in: {output_dir}')

if __name__ == '__main__':
    create_qampari_training_files()