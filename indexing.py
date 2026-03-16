from datasets import load_dataset
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
from colbert.data import Queries, Collection
import colbert

import argparse
import csv
from tqdm import tqdm
from pathlib import Path
import torch
import threading


import json
def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

def write_tsv(path, data):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

class GPUKeepAlive:
    """Runs lightweight GPU operations to keep GPU active during CPU work"""
    
    def __init__(self, interval=0.5, tensor_size=1024):
        self.interval = interval
        self.tensor_size = tensor_size
        self.running = False
        self.thread = None
        
        if torch.cuda.is_available():
            self.dummy_a = torch.randn(tensor_size, tensor_size, device='cuda')
            self.dummy_b = torch.randn(tensor_size, tensor_size, device='cuda')
            print(f"✓ GPUKeepAlive initialized ({tensor_size}x{tensor_size} tensors)", flush=True)
    
    def _worker(self):
        while self.running:
            if torch.cuda.is_available():
                # Aggressive continuous computation - no sleep!
                for _ in range(50):  # More iterations
                    _ = torch.matmul(self.dummy_a, self.dummy_b)
                    _ = torch.matmul(self.dummy_b, self.dummy_a)
                torch.cuda.synchronize()
    
    def start(self):
        if not torch.cuda.is_available() or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("✓ GPUKeepAlive: GPU now active during CPU operations", flush=True)
    
    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("✓ GPUKeepAlive stopped", flush=True)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_indexing', action='store_true')
    parser.add_argument('--do_retrieval', action='store_true')
    parser.add_argument('--do_processing', action='store_true')
    parser.add_argument('--do_training', action='store_true')
    parser.add_argument('--dataset', type=str, default='ambigqa')
    args = parser.parse_args()
    import time
    start_time = time.time()

    if args.do_processing:
        corpus_path = '/scratch/dq2024/diverse_retriever/chunks_v5.tsv'
        corpus = read_tsv(corpus_path)
        print(f'Loaded {len(corpus)} passages')
        for i, c in enumerate(tqdm(corpus)):
            if i == 0:
                continue
            corpus[i][0] = int(i-1)
            _ = corpus[i].pop()
            assert len(corpus[i]) == 2

        corpus.pop(0)

        write_tsv('/scratch/dq2024/wikipedia_chunks/colbert_chunks_v5.tsv', corpus)

    # #### Loading data ####
    # dataset = 'lifestyle'
    # datasplit = 'dev'

    # collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    # collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

    # queries_dataset = load_dataset("colbertv2/lotte", dataset)
    # queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

    # print(f'Loaded {len(queries)} queries and {len(collection):,} passages')
    # print(queries[24])
    # print()
    # print(collection[19929])
    # print()


    #### Indexing ####
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300 # truncate passages at 300 tokens
    # max_id = 10000

    # index_name = f'{dataset}.{datasplit}.{nbits}bits'

    # answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
    # filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]
    # print(f'Filtered down to {len(filtered_queries)} queries')
    # corpus_path = '/scratch/hc3337/wikipedia_chunks/colbert_chunks_v5.tsv'
    corpus_path = '/scratch/dq2024/wikipedia_chunks/colbert_chunks_v5.tsv'
    #corput_path = '/scratch/dq2024/diverse_retriever/chunks_v5.tsv'
    index_name = 'wikipedia.chunks_v5.2bits'

    if args.do_indexing:
        print('Indexing...')
        index_name = 'wikipedia.qampari_colbertv2.0_lr1e-5.2bits'
        checkpoint = 'colbert-ir/colbertv2.0'
        checkpoint = '/scratch/dq2024/ColBERT/experiments/qampari_colbertv2.0_lr1e-5/none/2026-03/04/18.34.25/checkpoints/colbert/'
        gpu_keepalive = GPUKeepAlive(interval=0.2, tensor_size=1024)
        with gpu_keepalive:
            with Run().context(RunConfig(nranks=1, experiment='wikipedia')):
                config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)
                indexer = Indexer(checkpoint=checkpoint, config=config)
                indexer.index(name=index_name, collection=corpus_path, overwrite=True)
        print(indexer.get_index())


    if args.do_retrieval:
        if args.dataset in ["ambigqa", "ambigqa_single", "qampari", "qampari_full"]:
            dataset_to_file = {"ambigqa": "ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl", "qampari": "qampari_dev_question_only_5_to_8_ctxs.jsonl", "ambigqa_single": "ambigqa_dev_single_answer_question_only.jsonl", "qampari_full": "dev_data_gt_qampari_corpus.jsonl"}
            data = read_jsonl(f'/scratch/dq2024/diverse_retriever/data/{dataset_to_file[args.dataset]}')
        elif args.dataset.endswith('.json'):
            data = read_json(args.dataset)
        else:
            data = read_jsonl(args.dataset)
        queries = [x['question_text'] if (args.dataset == 'qampari' or args.dataset == 'qampari_full') else x['question'] for x in data]
        outputs = []

        # To create the searcher using its relative name (i.e., not a full path), set
        # experiment=value_used_for_indexing in the RunConfig.
        # index_name = 'wikipedia.chunks_v5.2bits'
        index_name = 'wikipedia.qampari_colbertv2.0_lr1e-5.2bits'
        with Run().context(RunConfig(nranks=1, experiment='wikipedia')):
            config = ColBERTConfig(query_maxlen=512)
            searcher = Searcher(index=index_name, collection=corpus_path, config=config)

        for query in tqdm(queries):
            print(f"#> {query}")

            # Find the top-3 passages for this query
            results = searcher.search(query, k=100)

            # ['id', 'title', 'text', 'score']
            outputs.append({
                'question': query,
                'answers': [''],
                'ctxs': []
            })

            # Print out the top-k retrieved passages
            for passage_id, passage_rank, passage_score in zip(*results):
                print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

                outputs[-1]['ctxs'].append({
                    'id': passage_id,
                    'title': "",
                    'text': searcher.collection[passage_id],
                    'score': passage_score
                })


        if args.dataset in ["ambigqa", "ambigqa_single", "qampari", "qampari_full"]:
            output_path = f'/scratch/dq2024/ColBERT/outputs/{args.dataset}.jsonl'
        else:
            #output_path = f'/scratch/dq2024/ColBERT/outputs/' + str(Path(args.dataset).stem) + '_retrieved.jsonl'
            output_path = f'/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/colbert/retrieved_files/round2_retrieved.json'
        with open(output_path, 'w') as f:
            for output in outputs:
                f.write(json.dumps(output) + '\n')

        after_generating_embeddings = time.time()
        print(f"Time taken to retrieve: {after_generating_embeddings-start_time:.2f} seconds")

    if args.do_training:
        print(f'Starting ColBERTv2 Training on QAMPARI')
        
        with Run().context(RunConfig(nranks=2, experiment='qampari_colbertv2.0_lr1e-5')):
            config = ColBERTConfig(
                bsize=64,                       
                lr=1e-5,                      
                warmup=20_000,                 
                doc_maxlen=doc_maxlen,                
                dim=128,                       
                attend_to_mask_tokens=False,  
                nway=2,                        # Number of negatives (1 hard + 63 in-batch)
                accumsteps=1,                  
                similarity='cosine',           
                use_ib_negatives=True,          
                root="/scratch/dq2024/ColBERT/experiments"
            )
            
            print(f'Configuration:')
            print(f'  - Batch size: {config.bsize}')
            print(f'  - Learning rate: {config.lr}')
            print(f'  - Warmup steps: {config.warmup}')
            print(f'  - Doc maxlen: {config.doc_maxlen}')
            print(f'  - Embedding dim: {config.dim}')
            print(f'  - N-way (negatives): {config.nway}')
            print(f'  - In-batch negatives: {config.use_ib_negatives}')
            
            trainer = Trainer(
                triples="/scratch/dq2024/ColBERT/data/qampari_train_triples.jsonl",
                queries="/scratch/dq2024/ColBERT/data/qampari_train_queries.tsv",
                collection=corpus_path,  # Full Wikipedia corpus
                config=config,
            )
            
            print('Starting training from colbert-ir/colbertv2.0 checkpoint...\n')
            checkpoint_path = trainer.train(checkpoint='colbert-ir/colbertv2.0')
            
            print(f'Training Done. Saved checkpoint to: {checkpoint_path}')
