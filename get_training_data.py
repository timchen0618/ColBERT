import json
import csv
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
    # Load training data
    train_data = read_jsonl('/scratch/dq2024/diverse_retriever/train_data.jsonl')
    print(f'Loaded {len(train_data)} training examples')
    
    
    all_passages = {}  # {pid: text}
    
    print("Collecting all passages from training data...")
    for item in tqdm(train_data):
        # Collect from positive contexts
        for ctx in item.get('positive_ctxs', []):
            pid = ctx['id']
            text = ctx['text']
            all_passages[pid] = text
        
        # Collect from negative contexts
        for ctx in item.get('negative_ctxs', []):
            pid = ctx['id']
            text = ctx['text']
            all_passages[pid] = text
        
        # Collect from ground truths
        for ctx in item.get('ground_truths', []):
            pid = ctx['id']
            text = ctx['text']
            all_passages[pid] = text
        
        # Collect from hard negatives
        for ctx in item.get('hard_negative_ctxs', []):
            pid = ctx['id']
            text = ctx['text']
            all_passages[pid] = text
    
    print(f'Collected {len(all_passages)} unique passages')
    
    # Create collection.tsv
    collection_tsv = []
    for pid, text in all_passages.items():
        collection_tsv.append([pid, text])
    
    write_tsv('/scratch/hc3337/projects/ColBERT/data/qampari_train_collection.tsv', collection_tsv)
    print(f'Created collection.tsv with {len(collection_tsv)} passages')
    
    # Create queries.tsv
    queries_tsv = []
    for i, item in enumerate(tqdm(train_data, desc="Creating queries")):
        qid = i
        query_text = item['question_text']
        queries_tsv.append([str(qid), query_text])
    
    write_tsv('/scratch/hc3337/projects/ColBERT/data/qampari_train_queries.tsv', queries_tsv)
    print(f'Created queries.tsv with {len(queries_tsv)} queries')
    
    # Create triples.jsonl
    triples = []
    
    for i, item in enumerate(tqdm(train_data, desc="Creating triples")):
        qid = i
        
        # Get positive passage IDs
        positive_pids = []
        for pos_ctx in item.get('positive_ctxs', []):
            positive_pids.append(pos_ctx['id'])
        
        # Also include ground truths as positives
        for gt_ctx in item.get('ground_truths', []):
            pid = gt_ctx['id']
            if pid not in positive_pids:
                positive_pids.append(pid)
        
        # Get negative passage IDs
        negative_pids = []
        for neg_ctx in item.get('negative_ctxs', []):
            negative_pids.append(neg_ctx['id'])
        
        # Add hard negatives if available
        for hard_neg_ctx in item.get('hard_negative_ctxs', []):
            pid = hard_neg_ctx['id']
            if pid not in negative_pids:
                negative_pids.append(pid)
        
        # Create triples: [qid, positive_pid, negative_pid]
        if positive_pids and negative_pids:
            for pos_pid in positive_pids:
                # Sample up to 5 negatives per positive
                for neg_pid in negative_pids[:5]:
                    triples.append([qid, pos_pid, neg_pid])
        elif not positive_pids:
            print(f'Warning: Query {qid} has no positive passages')
        elif not negative_pids:
            print(f'Warning: Query {qid} has no negative passages')
    
    # Write triples to JSONL
    with open('/scratch/hc3337/projects/ColBERT/data/qampari_train_triples.jsonl', 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    
    print(f'Created triples.jsonl with {len(triples)} triples')
    print(f'Average triples per query: {len(triples) / len(train_data):.2f}')

if __name__ == '__main__':
    create_qampari_training_files()