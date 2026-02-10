from datasets import load_dataset
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import colbert

import argparse
import csv
from tqdm import tqdm

import json
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_indexing', action='store_true')
    parser.add_argument('--do_retrieval', action='store_true')
    parser.add_argument('--do_processing', action='store_true')
    parser.add_argument('--dataset', type=str, default='ambigqa')
    args = parser.parse_args()
    import time
    start_time = time.time()

    if args.do_processing:
        corpus_path = '/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv'
        corpus = read_tsv(corpus_path)
        print(f'Loaded {len(corpus)} passages')
        for i, c in enumerate(tqdm(corpus)):
            if i == 0:
                continue
            corpus[i][0] = int(i-1)
            _ = corpus[i].pop()
            assert len(corpus[i]) == 2

        corpus.pop(0)

        write_tsv('/scratch/hc3337/wikipedia_chunks/colbert_chunks_v5.tsv', corpus)

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
    corpus_path = '/scratch/hc3337/wikipedia_chunks/colbert_chunks_v5.tsv'
    index_name = 'wikipedia.chunks_v5.2bits'

    if args.do_indexing:
        print('Indexing...')
        checkpoint = 'colbert-ir/colbertv2.0'
        with Run().context(RunConfig(nranks=1, experiment='wikipedia')):  # nranks specifies the number of GPUs to use
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                        # Consider larger numbers for small datasets.
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=corpus_path, overwrite=True)
        print(indexer.get_index())


    if args.do_retrieval:
        dataset_to_file = {"ambigqa": "ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl", "qampari": "qampari_dev_question_only_5_to_8_ctxs.jsonl", "ambigqa_single": "ambigqa_dev_single_answer_question_only.jsonl"}
        data = read_jsonl(f'/scratch/hc3337/projects/autoregressive/data/questions/{dataset_to_file[args.dataset]}')
        queries = [x['question'] if args.dataset == 'ambigqa' or args.dataset == 'ambigqa_single' else x['question_text'] for x in data]
        outputs = []

        # To create the searcher using its relative name (i.e., not a full path), set
        # experiment=value_used_for_indexing in the RunConfig.
        with Run().context(RunConfig(experiment='wikipedia')):
            searcher = Searcher(index=index_name, collection=corpus_path)

        for query in tqdm(queries):
            print(f"#> {query}")

            # Find the top-3 passages for this query
            results = searcher.search(query, k=500)

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

        with open(f'/scratch/hc3337/projects/ColBERT/outputs/{args.dataset}.jsonl', 'w') as f:
            for output in outputs:
                f.write(json.dumps(output) + '\n')

        after_generating_embeddings = time.time()
        print(f"Time taken to retrieve: {after_generating_embeddings-start_time:.2f} seconds")
