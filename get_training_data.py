import json
import csv
import random
from tqdm import tqdm
from pathlib import Path

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

def write_tsv(path, data):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

def create_qampari_training_files():
    # Set random seed for reproducibility
    random.seed(42)
    
    train_data = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    print(f'Loaded {len(train_data)} training examples')
    
    # Create queries.tsv
    queries_tsv = []
    for i, item in enumerate(tqdm(train_data, desc="Creating queries")):
        qid = i
        query_text = item['question_text']
        queries_tsv.append([str(qid), query_text])
    
    output_dir = Path('/scratch/dq2024/ColBERT/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_tsv(output_dir / 'qampari_train_queries.tsv', queries_tsv)
    print(f'Created queries.tsv with {len(queries_tsv)} queries')
    
    # Create triples.jsonl
    # Format: [qid, positive_pid, negative_pid]
    # We randomly sample 1 negative per positive, in-batch negatives will provide more
    triples = []
    queries_without_data = 0
    
    for i, item in enumerate(tqdm(train_data, desc="Creating triples")):
        qid = i
        
        # Get positive passage IDs (positive_ctxs and ground_truths are the same)
        positive_pids = []
        for pos_ctx in item.get('positive_ctxs', []):
            pid = pos_ctx['id']
            if pid not in positive_pids:
                positive_pids.append(pid)
        
        # Get negative passage IDs (no hard negatives in your data)
        negative_pids = []
        for neg_ctx in item.get('negative_ctxs', []):
            pid = neg_ctx['id']
            if pid not in negative_pids:
                negative_pids.append(pid)
        
        # Create triples: randomly sample 1 negative per positive
        # In-batch negatives will provide additional negatives during training
        if positive_pids and negative_pids:
            for pos_pid in positive_pids:
                # Randomly sample one negative for this positive
                neg_pid = random.choice(negative_pids)
                triples.append([qid, pos_pid, neg_pid])
        else:
            queries_without_data += 1
            if not positive_pids:
                print(f'Warning: Query {qid} has no positive passages')
            if not negative_pids:
                print(f'Warning: Query {qid} has no negative passages')
    
    # Write triples to JSONL
    with open(output_dir / 'qampari_train_triples.jsonl', 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    
    print(f'\nCreated triples.jsonl with {len(triples)} triples')
    print(f'Queries without sufficient data: {queries_without_data}')
    print(f'Average triples per query: {len(triples) / len(train_data):.2f}')
    print(f'\nFiles created in: {output_dir}')
    print(f'  - qampari_train_queries.tsv')
    print(f'  - qampari_train_triples.jsonl')
    print(f'\nNote: Each positive passage is paired with a randomly sampled negative.')
    print(f'      In-batch negatives will provide {63} additional negatives during training.')

if __name__ == '__main__':
    create_qampari_training_files()