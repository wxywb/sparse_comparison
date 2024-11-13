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


def index_bm25(client, index_name, metadata):
    index_settings = {
        "settings": {
            "analysis": {"analyzer": {"default": {"type": "english"}}},
            "similarity": {"default": {"type": "BM25"}},
            "index.queries.cache.enabled": False  # Disable query cache
        },
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "english"},
                "doc_id": {"type": "keyword", "index": False},
                "chunk_id": {"type": "keyword", "index": False},
                "original_index": {"type": "integer", "index": False},
            }
        },
    }

    es_client.indices.create(index=index_name, body=index_settings)
    documents = metadata 
    actions = [
        {
            "_index": index_name,
            "_source": {
                "content": doc["content"],
                "doc_id": doc["doc_id"],
                "chunk_id": doc["chunk_id"],
                "original_index": doc["original_index"],
            },
        }
        for doc in documents
    ]
    success, _ = bulk(es_client, actions)
    es_client.indices.refresh(index=index_name)
    return success

def index_milvus(index_name, data):
    tokenizer_params = {
            "tokenizer": "standard",
            "filter":["lowercase", 
                {
                    "type": "length",
                    "max": 200,
                },{
                    "type": "stemmer",
                    "language": "english"
                },{
                    "type": "stop",
                    "stop_words": [
                        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", 
                        "no", "not", "of","how","what","where", "does","can", "do",  "on", "or", "such", "that", "the", "their", "then", "there", "these", 
                        "they", "this", "to", "was", "will", "with","I", "get"
                    ],
                }],
        }
        
    fields = [
        # Use auto generated id as primary key
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, analyzer_params=tokenizer_params, enable_match=True,  enable_analyzer=True),
        # We need a sparse vector field to perform full text search with BM25,
        # but you don't need to provide data for it when inserting data.
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="original_uuid", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="original_index", dtype=DataType.INT32),
    ]
    
    functions = [
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        )
    ]
    schema = CollectionSchema(fields, "", functions=functions)
    # Now we can create the new collection with above name and schema.
    col = Collection(index_name, schema, consistency_level="Strong")
    
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()
    
    embeddings = data["embeddings"]
    metadatas = data["metadata"]
    
    for embedding, metadata in zip(embeddings, metadatas):
        print(metadata)    
        col.insert({"dense_vector": embedding, **metadata})
    return col
 
if __name__ == '__main__':
    es_index_name = 'contextual_bm25_index4'
    es_client = Elasticsearch("http://localhost:9200")

    uri="http://10.102.6.57:19530"
    collection_name="milvus_hybrid_search4"
    connections.connect(uri = uri, collection_name= collection_name)

    db_path = "data/base_db/vector_db.pkl" 
    with open(db_path, "rb") as file:
        data = pickle.load(file)
    metadata = data["metadata"]

    index_milvus(collection_name, data)
    index_bm25(es_client, es_index_name, metadata)

