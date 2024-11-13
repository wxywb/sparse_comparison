import pickle
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import (
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
    connections,
    Function,
    FunctionType,
)

def load_jsonl(file_path: str):
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def bm25_delete_index(client, index_name):
    es_client.indices.delete(index=index_name)

if __name__ == '__main__':

    es_index_name = 'contextual_bm25_index4'
    es_client = Elasticsearch("http://localhost:9200")

    uri="http://10.102.6.57:19530"
    collection_name="milvus_hybrid_search4"
    connections.connect(uri = uri, collection_name= collection_name)

    col = Collection(collection_name)

    output_fields=[
            "content",
            "original_uuid",
            "doc_id",
            "chunk_id",
            "original_index",
    ]

    dataset = load_jsonl("evaluation_set.jsonl")

    k = 5

    total_bm25_query_score = 0
    total_milvus_query_score = 0
    num_queries = 0
    for query_item in dataset:
        query = query_item['query']
        golden_chunk_uuids = query_item['golden_chunk_uuids']
        
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
            if golden_doc:
                golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)
                if golden_chunk:
                    golden_contents.append(golden_chunk['content'].strip())

        es_client.indices.refresh(index=es_index_name)  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content"],
                }
            },
            "size": k,
        }

        bm25_response = es_client.search(index=es_index_name, body=search_body)
        bm25_results = [{"doc_id": hit["_source"]["doc_id"], "original_index": hit["_source"]["original_index"], "content": hit["_source"]["content"], "score": hit["_score"], } for hit in bm25_response["hits"]["hits"]]

        docs = col.search(data=[query], anns_field="sparse_vector", limit=k, output_fields=output_fields, param={})  
        milvus_results = [{'doc_id': doc.entity.doc_id, 'chunk_id': doc.entity.chunk_id, 'content': doc.entity.content} for doc in docs[0]]

        bm25_chunks_found = 0
        milvus_chunks_found = 0


        for golden_content in golden_contents:
            for doc in bm25_results[:k]:
                retrieved_content = doc['content'].strip()
                if retrieved_content == golden_content:
                    bm25_chunks_found += 1
                    break
        for golden_content in golden_contents:
            for doc in milvus_results[:k]:
                retrieved_content = doc['content'].strip()
                if retrieved_content == golden_content:
                    milvus_chunks_found += 1
                    break

        bm25_query_score = bm25_chunks_found / len(golden_contents)
        milvus_query_score = milvus_chunks_found / len(golden_contents)

        total_bm25_query_score += bm25_query_score
        total_milvus_query_score += milvus_query_score
        num_queries += 1
        print(total_bm25_query_score/num_queries, total_milvus_query_score/num_queries)


    


